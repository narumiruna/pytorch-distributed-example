# MNIST Example

Rank 0
```shell
$ python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=1234 \
    main.py
```

Rank 1
```shell
$ python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="127.0.0.1" \
    --master_port=1234 \
    main.py
```

## Run in docker

Install [docker](https://docs.docker.com/install/), [docker-compose](https://docs.docker.com/compose/install/) and [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) (if you want to run with GPU)

```
$ docker build --file Dockerfile --tag pytorch-distributed-example .
$ docker-compose up
For GPU
$ docker-compose --file docker-compose-gpu.yml up
```
