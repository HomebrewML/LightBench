# LightBench server/worker image tuned for single H100 (GPU 0) deployments
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0 \
    LIGHTBENCH_DEVICE=cuda:0

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_VERSION=2.4.0
ARG TORCHVISION_VERSION=0.19.0
ARG TORCHAUDIO_VERSION=2.4.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv git build-essential && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lightbench

COPY . /opt/lightbench

RUN python -m pip install --upgrade pip && \
    pip install --index-url ${PYTORCH_INDEX_URL} \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} && \
    pip install -e .

EXPOSE 8000

CMD ["uvicorn", "lightbench.web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
