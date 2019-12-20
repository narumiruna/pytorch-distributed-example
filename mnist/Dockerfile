FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

RUN pip install torchvision \
    && rm -rf ~/.cache/pip

ENV GLOO_SOCKET_IFNAME=eth0
ENV NCCL_SOCKET_IFNAME=eth0

WORKDIR /work
RUN python -c "from torchvision import datasets;datasets.MNIST('data', download=True)"
COPY main.py .
