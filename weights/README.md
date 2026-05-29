# Model weights

Place the StyleGAN2 fine-tuned checkpoint at this path:

```
weights/network-snapshot-001360.pkl
```

Specs:
- file size: ~283 MB
- sha256: (filled in at release time)
- format: standard `pickle` produced by NVlabs/stylegan3 `train.py` (cfg=stylegan2, cbase=16384)
- resolution: 256x256
- conditioning: unconditional

This file is hosted as an asset on the [GitHub Release](https://github.com/uniky98/dgm2026-face/releases)
because GitHub LFS quotas are small. Download with:

```bash
mkdir -p weights
wget -O weights/network-snapshot-001360.pkl \
    "https://github.com/uniky98/dgm2026-face/releases/download/v-final/network-snapshot-001360.pkl"
```

(or the equivalent `gh release download v-final --pattern network-snapshot-001360.pkl -D weights`.)

## Provenance

The weights are the result of:
1. Starting from `stylegan2-celebahq-256x256.pkl` (NVIDIA NGC; sha256
   `8b385aa564a58cdd97b4818620edcb27aeeb614345881b2ff0be91493b53fccc`).
2. Fine-tuning on `celebvhq_noalign256.zip` (285,328 PNGs, 256x256, no FFHQ
   alignment; see `report/final_report.pdf` §2 for the data pipeline) for
   2000 kimg with `cfg=stylegan2 batch=32 gamma=8 mirror=1 aug=ada target=0.7
   cbase=16384` on a single H200.
3. Selecting the kimg 1360 snapshot as our best by per-snap fid50k_full
   (lowest at FID 13.26) and by leaderboard FID (34.96) after sampling at
   psi=0.9.
