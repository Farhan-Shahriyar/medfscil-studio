"""
Build JPEG thumbnail packs for the images the demo displays.

The source .npz members are deflate-compressed, so fetching image #4321 means streaming
everything before it. That is acceptable once, offline; it is not acceptable on every
click in a UI. Each pack stores JPEG bytes end to end plus an offset table, so a lookup
is a slice and a ~1 ms decode.

Two kinds of pack are needed:
  test  - every test image, for error galleries and the image browser
  train - ONLY the novel classes, because those are the only images that can ever be
          drawn as K-shot support. Base classes always use their full training mean,
          so their individual images are never shown. On PathMNIST that is 35k images
          instead of 90k.

  python build_thumbs.py --split test  --size 160
  python build_thumbs.py --split train --size 128 --novel-only
"""
import argparse, io, os, zipfile
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('MEDMNIST_DIR') or os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'thumbs')
N_BASE = {'dermamnist': 4, 'pathmnist': 5, 'bloodmnist': 4, 'retinamnist': 3}


def read_header(fh):
    ver = np.lib.format.read_magic(fh)
    return (np.lib.format.read_array_header_1_0(fh) if ver == (1, 0)
            else np.lib.format.read_array_header_2_0(fh))


def load_small(zf, name):
    with zf.open(name) as fh:
        shape, _, dt = read_header(fh)
        buf = fh.read()
    return np.frombuffer(buf, dtype=dt).reshape(shape)


def stream(zf, name, batch=64):
    with zf.open(name) as fh:
        shape, _, dt = read_header(fh)
        n = shape[0]
        item = int(np.prod(shape[1:])) * np.dtype(dt).itemsize
        done = 0
        while done < n:
            b = min(batch, n - done)
            buf = fh.read(item * b)
            yield done, np.frombuffer(buf, dtype=dt).reshape((b,) + tuple(shape[1:]))
            done += b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=160)
    ap.add_argument('--quality', type=int, default=88)
    ap.add_argument('--split', default='test')
    ap.add_argument('--novel-only', action='store_true',
                    help='keep only classes that arrive as novel sessions (train packs)')
    ap.add_argument('--datasets', default=','.join(N_BASE))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for d in a.datasets.split(','):
        dst = os.path.join(OUT, f'{d}_{a.split}_thumbs.npz')
        if os.path.exists(dst) and not a.force:
            print(f'{d}/{a.split}: cache hit'); continue

        zf = zipfile.ZipFile(os.path.join(DATA_DIR, f'{d}_224.npz'))
        labels = load_small(zf, f'{a.split}_labels.npy').reshape(-1).astype(np.int64)

        if a.novel_only:
            counts = np.bincount(labels, minlength=int(labels.max()) + 1)
            # class order must match the FSCIL protocol: by TRAIN frequency, descending
            tr_labels = load_small(zf, 'train_labels.npy').reshape(-1).astype(np.int64)
            tr_counts = np.bincount(tr_labels, minlength=len(counts))
            order = np.argsort(-tr_counts).tolist()
            keep_cls = set(order[N_BASE[d]:])
            wanted = np.where(np.isin(labels, list(keep_cls)))[0]
        else:
            wanted = np.arange(len(labels))

        want_set = set(int(i) for i in wanted)
        blobs = []
        for start, chunk in stream(zf, f'{a.split}_images.npy'):
            for k, img in enumerate(chunk):
                gi = start + k
                if gi not in want_set:
                    continue
                im = Image.fromarray(img)
                if a.size and a.size != im.width:
                    im = im.resize((a.size, a.size), Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, 'JPEG', quality=a.quality, optimize=True)
                blobs.append(np.frombuffer(buf.getvalue(), dtype=np.uint8))

        offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
        flat = np.concatenate(blobs) if blobs else np.zeros(0, np.uint8)
        np.savez(dst, data=flat, offsets=offsets, index=wanted.astype(np.int64))
        print(f'{d}/{a.split}: {len(blobs)} thumbs at {a.size}px, {flat.nbytes/1e6:.1f} MB'
              + ('  (novel classes only)' if a.novel_only else ''))


if __name__ == '__main__':
    main()
