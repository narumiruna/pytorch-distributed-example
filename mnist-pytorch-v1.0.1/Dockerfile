FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

RUN pip install torchvision \
    && rm -rf ~/.cache/pip

ENV GLOO_SOCKET_IFNAME=eth0

WORKDIR /work
COPY main.py .