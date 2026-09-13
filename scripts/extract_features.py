"""
Extract frozen features from every backbone in the study, for every MedMNIST-224 dataset.

Backbones (all frozen, exactly as the benchmark notebooks use them):
  resnet50    torchvision ResNet-50, IMAGENET1K_V2, 2048-d GAP
  dinov2      facebook/dinov2-base (the fusion notebook's ungated fallback for DINOv3),
              CLS + patch-GAP concatenated = 1536-d
  biomedclip  microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 visual encoder, 512-d

The source .npz members are deflate-compressed, so images are streamed sequentially
rather than loaded whole - PathMNIST alone is 12.6 GB.

Every dataset is already stored at 224x224, so each backbone's Resize/CenterCrop is a
no-op and the transform reduces to ToTensor() + Normalize(mean, std). We therefore read
the backbone's own normalisation constants and apply them directly to the raw uint8
stream, which is pixel-identical to running its torchvision transform on a PIL image.

  python extract_features.py --backbone resnet50 --datasets dermamnist,pathmnist
"""
import argparse, os, sys, time, zipfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Where the MedMNIST-224 .npz archives live. Override with the MEDMNIST_DIR environment
# variable; otherwise a `data/` folder beside this project is assumed.
DATA_DIR = os.environ.get('MEDMNIST_DIR') or os.path.join(HERE, 'data')
ALL_DATASETS = ['dermamnist', 'pathmnist', 'bloodmnist', 'retinamnist']
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


# ----------------------------------------------------------------- npz streaming
def read_header(fh):
    ver = np.lib.format.read_magic(fh)
    return (np.lib.format.read_array_header_1_0(fh) if ver == (1, 0)
            else np.lib.format.read_array_header_2_0(fh))


def load_small(zf, name):
    with zf.open(name) as fh:
        shape, _, dt = read_header(fh)
        buf = fh.read()
    return np.frombuffer(buf, dtype=dt).reshape(shape)


def stream_images(zf, name, batch):
    with zf.open(name) as fh:
        shape, _, dt = read_header(fh)
        n = shape[0]
        item = int(np.prod(shape[1:])) * np.dtype(dt).itemsize
        done = 0
        while done < n:
            b = min(batch, n - done)
            buf = fh.read(item * b)
            if len(buf) != item * b:
                raise IOError(f'short read on {name}')
            yield np.frombuffer(buf, dtype=dt).reshape((b,) + tuple(shape[1:]))
            done += b


# ----------------------------------------------------------------- backbones
def build_resnet50(device):
    import torchvision.models as models
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    m = m.to(device).eval()
    return (lambda x: m(x)), IMAGENET, 2048, m


def build_dinov2(device, model_id='facebook/dinov2-base'):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(model_id).to(device).eval()
    nreg = getattr(m.config, 'num_register_tokens', 0)
    dim = m.config.hidden_size

    def fwd(x):
        seq = m(pixel_values=x).last_hidden_state
        cls = seq[:, 0]
        gap = seq[:, 1 + nreg:].mean(dim=1)
        return torch.cat([cls, gap], dim=1)          # CLS + GAP, as the fusion notebook
    return fwd, IMAGENET, dim * 2, m


def build_biomedclip(device):
    import open_clip
    m, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    m = m.to(device).eval()
    # pull the real normalisation constants out of the model's own transform
    mean = std = None
    for t in getattr(preprocess, 'transforms', []):
        if t.__class__.__name__ == 'Normalize':
            mean, std = tuple(t.mean), tuple(t.std)
    if mean is None:
        mean, std = ((0.48145466, 0.4578275, 0.40821073),
                     (0.26862954, 0.26130258, 0.27577711))
    with torch.no_grad():
        dim = m.encode_image(torch.zeros(1, 3, 224, 224, device=device)).shape[1]
    return (lambda x: m.encode_image(x)), (mean, std), dim, m


BUILDERS = {'resnet50': build_resnet50, 'dinov2': build_dinov2, 'biomedclip': build_biomedclip}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', required=True, choices=list(BUILDERS))
    ap.add_argument('--datasets', default=','.join(ALL_DATASETS))
    ap.add_argument('--splits', default='train,test')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'features'))
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = os.path.join(a.out, a.backbone)
    os.makedirs(out_dir, exist_ok=True)

    fwd, (mean, std), dim, model = BUILDERS[a.backbone](device)
    for p in model.parameters():
        p.requires_grad = False
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    print(f'[{a.backbone}] device={device} dim={dim} mean={tuple(round(float(v),4) for v in mean)}',
          flush=True)

    for dname in a.datasets.split(','):
        zf = zipfile.ZipFile(os.path.join(DATA_DIR, f'{dname}_224.npz'))
        for split in a.splits.split(','):
            fo = os.path.join(out_dir, f'{dname}_{split}_feats.npy')
            lo = os.path.join(out_dir, f'{dname}_{split}_labels.npy')
            labels = load_small(zf, f'{split}_labels.npy').reshape(-1).astype(np.int64)
            np.save(lo, labels)
            if os.path.exists(fo):
                print(f'  [{dname}/{split}] cache hit {np.load(fo, mmap_mode="r").shape}', flush=True)
                continue

            n = len(labels)
            feats = np.empty((n, dim), dtype=np.float32)
            done, t0 = 0, time.time()
            with torch.no_grad():
                for chunk in stream_images(zf, f'{split}_images.npy', a.batch):
                    x = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)
                    x = x.permute(0, 3, 1, 2).float().div_(255.0)
                    x = (x - mean_t) / std_t
                    with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                        f = fwd(x)
                    feats[done:done + len(chunk)] = f.float().cpu().numpy()
                    done += len(chunk)
                    if done % (a.batch * 60) == 0 or done == n:
                        el = time.time() - t0
                        print(f'  [{dname}/{split}] {done}/{n} {done/el:.0f} img/s '
                              f'eta {(n-done)/(done/el):.0f}s', flush=True)
            np.save(fo, feats)
            print(f'  [{dname}/{split}] saved {feats.shape} in {time.time()-t0:.0f}s', flush=True)
    print(f'[{a.backbone}] done', flush=True)


if __name__ == '__main__':
    main()
