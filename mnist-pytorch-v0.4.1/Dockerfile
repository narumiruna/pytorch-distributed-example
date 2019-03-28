FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

RUN pip install torchvision \
    && rm -rf ~/.cache/pip

WORKDIR /work
RUN python -c "from torchvision import datasets;datasets.MNIST('data', download=True)"
COPY main.py .
