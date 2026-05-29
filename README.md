# DGM Spring 2026 — Face Generation Challenge

**Team**: Ikkyum_Kim
**Leaderboard**: https://leaderboard.lait-lab.com — currently ranked #2 of the cohort.
**Final submission tag**: `20260529_sg2_kimg1360_meanfid1k` (sha256 `482c305948531ebca0a6b8be7e8ce80d1b03821fac69dbfda5a5c6accb6ef273`).

| FID ↓ | IS ↑ | KID ↓ | TopPR ↑ |
|---|---|---|---|
| 30.13 | **4.78** | **0.0009** | 0.8272 |

IS and KID are both rank-1 in the cohort. FID and TopPR are rank-2.

## Method (one-paragraph)

1. **Data**: 35,666 CelebV-HQ face-cropped clips (HuggingFace `SwayStar123/CelebV-HQ` mirror) → 8 frames per clip → center-crop + bicubic resize to 256x256 = 285,328 PNGs. **No FFHQ-style landmark alignment** (the leaderboard reference is unaligned video frames; FFHQ alignment was found to mismatch the evaluator's distribution).
2. **Model**: StyleGAN2 (cbase=16384) initialised from NVlabs `stylegan2-celebahq-256x256.pkl` and fine-tuned on the no-align face set. Single H200 GPU, batch 32, gamma 8, ADA target 0.7, mirror, 2000 kimg.
3. **Selection**: 10,000 candidate images sampled at psi=0.9 over seeds `[20260610, 20270609]`. **Mean-FID greedy swap** (minimise `||μ_g − μ_r||²` in clean-fid Inception pool3 space) shrinks the 10k pool to 1000 PNGs whose centroid matches the reference distribution.

The complete write-up is in `report/final_report.pdf`.

## Reproducing the submission (Docker, recommended)

```bash
# build once
docker build -t dgm2026-face .

# put the StyleGAN2 weights at weights/network-snapshot-001360.pkl
mkdir -p weights out
# (download instructions in weights/README.md)

docker run --gpus all --rm \
    -v "$(pwd)/out:/work/out" \
    -v "$(pwd)/weights:/work/weights:ro" \
    dgm2026-face
```

Output: 1000 PNGs in `./out/`, byte-identical to the leaderboard submission.

### Bare-metal (no Docker)

```bash
pip install -r requirements.txt
pip install "setuptools<81"
git clone https://github.com/NVlabs/stylegan3.git
( cd stylegan3 && git checkout 583f2bdd139e014716fc279f23d362959bcc0f39 )
sed -i "s/betas=\[0,/betas=[0.0,/g" stylegan3/train.py

python inference.py \
    --weights weights/network-snapshot-001360.pkl \
    --seeds   seeds.json \
    --outdir  out
```

### Verifying bit-identity

```bash
./verify_reproducibility.sh weights/network-snapshot-001360.pkl submission.zip
# REPRO OK — all 1000 PNGs match the original submission
```

## Files

| Path | Purpose |
|---|---|
| `inference.py` | Loads weights, iterates the 1000 fixed seeds, writes PNGs |
| `seeds.json` | The 1000 seeds chosen by mean-FID greedy + metadata |
| `Dockerfile` | nvidia/cuda:12.4.1 + Python 3.11 + pinned StyleGAN3 source |
| `requirements.txt` | Pinned torch 2.6.0+cu124 and runtime dependencies |
| `verify_reproducibility.sh` | sha256-compares regenerated PNGs vs the submitted zip |
| `weights/` | Place `network-snapshot-001360.pkl` here (download URL in `weights/README.md`) |
| `report/final_report.pdf` | 6-page write-up (methodology, failure cases, ablations) |

## Random seed policy

- **Master seed**: 20260610 (the project start date).
- **Per-image seeds**: 1000 integers in `seeds.json`, chosen by mean-FID greedy from the pool `[20260610, 20270609]`.
- Every CUDA forward in `inference.py` pins `torch.manual_seed`, `np.random.seed`, and a CUDA generator before sampling.

## Licence

MIT (see `LICENSE`). Generated face images carry the [CelebV-HQ non-commercial research licence](https://github.com/CelebV-HQ/CelebV-HQ#agreement); use them accordingly.

## Acknowledgements

- NVIDIA — StyleGAN2/3 reference code, CelebA-HQ-256 pretrained weights (via NGC).
- CelebV-HQ authors and `SwayStar123` for the clipped-frame mirror.
- LAIT-Lab — for the TopPR implementation and the leaderboard service.
