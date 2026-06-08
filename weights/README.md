# Model weights

Place the StyleGAN2 fine-tuned checkpoint at this path:

```
weights/network-snapshot-002720.pkl
```

Specs:
- file size: ~283 MB (295,763,991 bytes)
- sha256: 6e241d9b143e209df92b4b37e2fc65b8921a94a8d07221a5d4927f9c5ec2337d
- format: standard `pickle` produced by NVlabs/stylegan3 `train.py` (cfg=stylegan2, cbase=16384)
- resolution: 256x256
- conditioning: unconditional

This file is hosted as an asset on the [GitHub Release](https://github.com/uniky98/dgm2026-face/releases)
because GitHub LFS quotas are small. Download with:

```bash
mkdir -p weights
wget -O weights/network-snapshot-002720.pkl \
    "https://github.com/uniky98/dgm2026-face/releases/download/v-final/network-snapshot-002720.pkl"
```

(or the equivalent `gh release download v-final --pattern network-snapshot-002720.pkl -D weights`.)

## Provenance

The weights are the result of:
1. Starting from `stylegan2-celebahq-256x256.pkl` (NVIDIA NGC; sha256
   `8b385aa564a58cdd97b4818620edcb27aeeb614345881b2ff0be91493b53fccc`).
2. Fine-tuning on `celebvhq_noalign256.zip` (285,328 PNGs, 256x256, no FFHQ
   alignment; see `report/` §2 for the data pipeline) with
   `cfg=stylegan2 batch=32 gamma=8 mirror=1 aug=ada target=0.7 cbase=16384`
   on a single H200, in two phases: an initial run, then a continuation
   resumed from the kimg-1360 checkpoint.
3. Selecting the continuation's kimg-2720 snapshot as our best by per-snap
   fid50k_full (11.867, improving on the kimg-1360 snapshot's 13.26). Sampled
   at psi=0.9 and reduced to 1,000 images by mean-FID greedy selection, this
   checkpoint scores leaderboard FID 29.06 / IS 5.07 / KID 0.0009 /
   TopPR 0.8441 (rank #1 by average rank).
