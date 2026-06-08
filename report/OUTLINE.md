# Final Challenge Report — Outline (6 pages, NeurIPS/ICLR-style)

Target: 6 pages incl. references. Use the Group Final Report template.

## §1 Problem & evaluation (≤ 0.5 p)
- CelebV-HQ 32,550-frame reference set, 4-metric combined ranking
  (FID, IS, KID, TopPR; AVG-RANK).
- Submission contract (≤200 MB zip, 1000 PNG, ≤5 GB weights, reproducibility).

## §2 Data pipeline (≤ 1 p)
- CelebV-HQ acquisition via SwayStar123 HF mirror (avoids the dead-YouTube-link
  attrition of the official downloader).
- Frame sampling: 8 evenly-spaced frames per clip → 285,328 raw frames.
- **Critical decision**: FFHQ-style 5-point alignment WAS used initially and was
  shown to *hurt* (Phase A in §4); we switched to bbox crop + bicubic 256
  resize.
- Reference for local FID/KID calibration: random 32,550 subset (seed 0) of our
  noalign 285k. Matches leaderboard within 4 decimal places — proves the
  evaluator's reference IS bbox-cropped, unaligned video frames.

## §3 Model selection (≤ 1 p)
- StyleGAN3-T-FFHQ-U-256 first (Phase A): leaderboard FID 55, plateaued.
- StyleGAN2-CelebA-HQ-256 (Phase B): same architecture family but pretrained on
  the *celebrity photo* distribution that overlaps CelebV-HQ identities
  → leaderboard FID 36 in 16 h.
- Why not diffusion (EDM2/SD 1.5)? **SD 1.5 with naive prompts was tried**
  and produced studio-portrait distribution mismatch (FID 97). EDM2-img64
  ckpts are all class-conditional ImageNet — heavy adaptation cost.

## §4 Failure cases (extra credit — ≤ 1.5 p)
*The single most-marked section per the update; rigorous treatment expected.*

### 4.1 FFHQ-aligned dataset (Phase A)
- Internal FID 7.28 ← **looks** like a SOTA face GAN
- Leaderboard FID 55.28, KID 0.0227 ← 7.5× and 13× gaps vs internal
- Root cause: 50k-vs-50k FID metric ≠ 1k-vs-32550 metric *and* our train data
  distribution was over-aligned relative to evaluator's natural video frames.

### 4.2 SD 1.5 "stronger prior" hypothesis
- Pretrained SD 1.5 + 26 diverse face prompts → leaderboard FID 96.87,
  TopPR 0.6221, IS 4.55.
- Failure mode: prompt-induced studio-portrait distribution sits in a
  completely different mode of InceptionV3 feature space than CelebV-HQ
  video frames.

### 4.3 Centroid-distance "oracle" filter
- Greedy pick 1000 closest to reference centroid in InceptionV3 feature
  space → produces mode collapse, FID 35→58, KID 0.007→0.029.
- Insight: minimising mean-distance per-image without preserving the
  set's covariance term *destroys* the second-moment match that FID also
  measures.

### 4.4 Multi-ψ and multi-ckpt sample mixes
- Both made FID worse (3-4 unit increase). Implication: mixing distributions
  inflates the InceptionV3 covariance more than it improves coverage.

## §5 Final method — mean-FID greedy + bbox-noalign + SG2-CelebA-HQ (≤ 1.5 p)

### Selection algorithm
```
init random 1000 from 10k pool
repeat 4 passes, random order over the 1000:
  for each in-set item k:
    delta = pool[c] - pool[old]
    new_d2 = cur_d2 + (2/n) v.delta + (1/n²) ||delta||²    # v = μ_g - μ_r
    swap to argmin c if it decreases cur_d2
```
- O(N · M · d) per pass; ~7 min total on 10k×2048 features.
- *Key difference vs centroid oracle*: we **swap**, not **pick**, so the
  pool's spread is preserved → covariance term stays small → full FID drops.

### Results (local 4-metric vs leaderboard, single submission)
| | FID | KID | IS | TopPR |
|---|---|---|---|---|
| local 32550 seed 0 (kimg-1360) | 29.60 | 0.000354 | 4.65 | 0.78 (lib f1) |
| leaderboard (kimg-1360, mean-FID) | 30.13 | 0.0009 | 4.78 | 0.8272 |
| **leaderboard (final, kimg-2720 cont.)** | **29.06** | **0.0009** | **5.07** | **0.8441** |
| 1st place (Sangwon) | 28.97 | 0.0017 | 4.66 | 0.8368 |

→ **IS and KID rank-1 in the cohort.** FID rank-2 (by 0.09); the continuation
checkpoint lifts TopPR from rank ~9 to rank ~6 → **#1 overall by average rank.**

## §6 Ablations & sensitivity (≤ 0.5 p)
- Pool size: 10k pool ≫ random 1k baseline (FID 35.12 vs 29.60).
- Passes 1→4: FID converges by pass 2-3 (98%+ of swaps in pass 1).
- Ref subset seed: FID 29.58 – 29.72 over 5 seeds (negligible).
- Truncation ψ on the *underlying* sampler: 0.85 < 0.9 < 1.0 in IS but
  0.9 best in FID.

## §7 Compute & reproducibility (≤ 0.5 p)
- Total wall-clock: ~24 h GPU on a shared H200 (kept conservative at batch 32
  to coexist with another user's `totem` workload).
- All steps are scripted; `verify_reproducibility.sh` proves bit-identity.
- GitHub: `uniky98/dgm2026-face`. Weights: GitHub Release `v-final`.

## §8 References & code attribution
- StyleGAN2-ADA / StyleGAN3 — NVlabs.
- Clean-FID (Parmar et al. 2022) — used unchanged.
- TopPR — LAIT-CVLab (also the leaderboard provider).
- CelebV-HQ — original authors; data via SwayStar123 mirror.
