#!/usr/bin/env python
"""Generate the two report figures:
  samples.jpg     - 5x5 montage of generated faces from the final submission
  calibration.pdf - local-vs-leaderboard FID scatter + KID over-optimism bars
Run from the report/ directory:  python make_figs.py
"""
import io, zipfile
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ZIP = "../../../submissions/20260530_sg2_newckpt_meanfid1k/submission.zip"

# ---------- Figure 1: sample faces montage (5x5) ----------
z = zipfile.ZipFile(ZIP)
names = sorted((n for n in z.namelist() if n.endswith(".png")),
               key=lambda n: int(n.replace("seed", "").replace(".png", "")))
N, G, T, P = 25, 5, 200, 6          # 25 tiles, 5x5 grid, 200px tile, 6px pad
idx = [round(i * (len(names) - 1) / (N - 1)) for i in range(N)]
canvas = Image.new("RGB", (G * T + (G + 1) * P,) * 2, (255, 255, 255))
for k, j in enumerate(idx):
    im = Image.open(io.BytesIO(z.read(names[j]))).convert("RGB").resize((T, T))
    r, c = divmod(k, G)
    canvas.paste(im, (P + c * (T + P), P + r * (T + P)))
canvas.save("samples.jpg", quality=92)
print("wrote samples.jpg", canvas.size)

# ---------- Figure 2: calibration ----------
# (local FID, leaderboard FID) per submission
fid = {
    "random-1k\n(baseline)":      (35.12, 34.96),
    "kimg-1360\nmean-FID":        (29.60, 30.13),
    "kimg-2720\nmean-FID (A)":    (28.53, 29.06),
    "kimg-1360\n30k-pool (B)":    (28.97, 29.50),
}
# (local KID, leaderboard KID) x1e-3 for the three mean-FID submissions
kid_local = np.array([0.000354, 0.000273, 0.0000213]) * 1e3
kid_lb    = np.array([0.0009,   0.0009,   0.0006])    * 1e3
kid_labels = ["kimg-1360", "kimg-2720 (A)", "30k-pool (B)"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))

# left: FID local vs leaderboard
xs = np.array([v[0] for v in fid.values()])
ys = np.array([v[1] for v in fid.values()])
lo, hi = 27.8, 35.8
ax1.plot([lo, hi], [lo, hi], "--", color="0.6", lw=1, label="y = x")
ax1.plot([lo, hi], [lo + 0.53, hi + 0.53], ":", color="C3", lw=1.2,
         label="y = x + 0.53 (mean-FID regime)")
for (lab, (x, y)), col in zip(fid.items(), ["0.4", "C0", "C2", "C1"]):
    ax1.scatter([x], [y], s=42, color=col, zorder=3)
    ax1.annotate(lab, (x, y), fontsize=6.5, xytext=(4, -10),
                 textcoords="offset points")
ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
ax1.set_xlabel("local FID (proxy ref, seed 0)")
ax1.set_ylabel("leaderboard FID")
ax1.set_title("(a) FID is predictable: +0.53 offset", fontsize=9)
ax1.legend(fontsize=6.5, loc="upper left")
ax1.grid(alpha=0.25)

# right: KID over-optimism bars
xb = np.arange(len(kid_labels)); w = 0.36
ax2.bar(xb - w/2, kid_local, w, label="local KID", color="C0")
ax2.bar(xb + w/2, kid_lb,    w, label="leaderboard KID", color="C3")
ax2.set_xticks(xb); ax2.set_xticklabels(kid_labels, fontsize=7)
ax2.set_ylabel(r"KID ($\times 10^{-3}$)")
ax2.set_title("(b) KID is over-optimistic locally", fontsize=9)
ax2.legend(fontsize=7)
ax2.grid(alpha=0.25, axis="y")

fig.tight_layout()
fig.savefig("calibration.pdf")
print("wrote calibration.pdf")
