#!/usr/bin/env python
"""Inference for DGM Spring 2026 Face Generation Challenge — Team Ikkyum_Kim.

Bit-identical reproduction of the 1000-image submission.

Pipeline:
  1. Load the fine-tuned StyleGAN2 checkpoint
     (SG2-CelebA-HQ-256 → CelebV-HQ noalign, kimg 1360, psi=0.9).
  2. Read the 1000 selection seeds from seeds.json. Those seeds are the
     output of mean-FID-greedy applied to the 10,000-seed candidate pool
     [20260610, 20270609].
  3. Generate one 256x256 PNG per seed and write to --outdir.

The generation pipeline mirrors NVlabs/stylegan3 `gen_images.py` byte-for-byte
so that `verify_reproducibility.sh` succeeds against the original submission.

Usage:
    python inference.py --weights weights/network-snapshot-001360.pkl \\
                        --seeds   seeds.json \\
                        --outdir  out
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to network-snapshot-001360.pkl")
    ap.add_argument("--seeds",   required=True, help="path to seeds.json")
    ap.add_argument("--outdir",  required=True, help="output dir for PNGs")
    ap.add_argument("--trunc",   type=float, default=0.9, help="truncation psi")
    ap.add_argument("--noise-mode", default="const", choices=["const", "random", "none"])
    ap.add_argument("--stylegan3-repo", default="stylegan3",
                    help="path to NVlabs/stylegan3 source tree")
    args = ap.parse_args()

    # Make bundled NVlabs/stylegan3 importable. The repo supports the stylegan2
    # config (cfg=stylegan2), which is what our fine-tune is.
    sys.path.insert(0, args.stylegan3_repo)
    import dnnlib              # noqa: E402
    import legacy              # noqa: E402

    print(f"[inference] loading network: {args.weights}", flush=True)
    device = torch.device("cuda")
    with dnnlib.util.open_url(args.weights) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    print(f"[inference] reading seed list: {args.seeds}", flush=True)
    with open(args.seeds) as f:
        meta = json.load(f)
    seeds = meta["seeds"]
    assert len(seeds) == 1000, f"expected 1000 seeds, got {len(seeds)}"

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # Unconditional generator (c_dim == 0). Label is an empty tensor.
    label = torch.zeros([1, G.c_dim], device=device)

    # Matches NVlabs gen_images.py exactly so that sha256 comparison passes.
    print(f"[inference] generating {len(seeds)} images "
          f"(psi={args.trunc}, noise_mode={args.noise_mode})", flush=True)
    for i, seed in enumerate(seeds):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=args.trunc, noise_mode=args.noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(out / f"seed{seed:04d}.png")
        if (i + 1) % 100 == 0 or i + 1 == len(seeds):
            print(f"  {i+1}/{len(seeds)}", flush=True)

    print(f"[inference] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
