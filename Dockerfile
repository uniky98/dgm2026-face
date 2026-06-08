# DGM Spring 2026 Face Generation Challenge — Team Ikkyum_Kim
# One-command reproduction of the submission images.
#
# Build:
#   docker build -t dgm2026-face .
# Run (with NVIDIA GPU):
#   docker run --gpus all --rm -v $(pwd)/out:/work/out -v $(pwd)/weights:/work/weights:ro \
#       dgm2026-face
# Output: 1000 PNGs in ./out matching the leaderboard submission bit-for-bit.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System packages: Python 3.11, git (for pinned NVlabs/stylegan3), build tools
# for StyleGAN2 custom CUDA ops, ninja + nvcc-compatible gcc 13.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
        python3-pip \
        git wget ca-certificates \
        gcc-13 g++-13 \
        ninja-build \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Python dependencies (cu124 torch matches host driver 535/580 forward).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pin NVlabs/stylegan3 to the commit we developed against.
RUN git clone https://github.com/NVlabs/stylegan3.git \
 && cd stylegan3 \
 && git checkout 583f2bdd139e014716fc279f23d362959bcc0f39

# Apply the same compatibility patches we apply on our host:
# 1) `betas=[0, 0.99]` -> `betas=[0.0, 0.99]` so torch 2.x Adam accepts list[float].
RUN sed -i "s/betas=\[0,/betas=[0.0,/g" stylegan3/train.py
# 2) Newer setuptools removed pkg_resources at runtime; SG3 imports it.
RUN pip install --no-cache-dir "setuptools<81"

COPY inference.py seeds.json verify_reproducibility.sh ./
RUN chmod +x verify_reproducibility.sh

ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"

# Default: regenerate the 1000 submission images.
CMD ["python", "inference.py", \
     "--weights", "weights/network-snapshot-002720.pkl", \
     "--seeds",   "seeds.json", \
     "--outdir",  "out", \
     "--stylegan3-repo", "stylegan3"]
