"""Thumbnail access for the UI, backed by the JPEG packs from scripts/build_thumbs.py.

Packs may be partial (the train packs hold only novel-class images), so every pack
carries the original dataset indices it covers and lookups go through that map.
"""
import io, os
import numpy as np
from PIL import Image

from .config import THUMBS


class ThumbPack:
    def __init__(self, root=THUMBS):
        self.root = root
        self._packs = {}

    def _pack(self, dataset, split):
        key = (dataset, split)
        if key not in self._packs:
            p = os.path.join(self.root, f'{dataset}_{split}_thumbs.npz')
            if not os.path.exists(p):
                self._packs[key] = None
            else:
                z = np.load(p)
                idx = z['index'] if 'index' in z.files else np.arange(len(z['offsets']) - 1)
                self._packs[key] = (z['data'], z['offsets'],
                                    {int(v): i for i, v in enumerate(idx)})
        return self._packs[key]

    def available(self, dataset, split='test'):
        return self._pack(dataset, split) is not None

    def get(self, dataset, idx, split='test'):
        pk = self._pack(dataset, split)
        if pk is None:
            return None
        data, off, imap = pk
        slot = imap.get(int(idx))
        if slot is None:
            return None
        return Image.open(io.BytesIO(data[off[slot]:off[slot + 1]].tobytes())).convert('RGB')

    def many(self, dataset, indices, split='test'):
        out = []
        for i in indices:
            im = self.get(dataset, int(i), split)
            if im is not None:
                out.append(im)
        return out


THUMBS_PACK = ThumbPack()
