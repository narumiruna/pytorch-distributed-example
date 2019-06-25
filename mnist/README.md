# MNIST Example

```shell
export GLOO_SOCKET_IFNAME=eth0
```

Rank 0
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 2
```

Rank 1
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 2
```

## Use specific root directory for running example on single machine.

Rank 0
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 2 --root data0
```

Rank 1
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 2 --root data1
```

## Run in docker

Install [docker](https://docs.docker.com/install/), [docker-compose](https://docs.docker.com/compose/install/) and [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) (if you want to run with GPU)

```
$ docker build --file Dockerfile --tag pytorch-distributed-example .
$ docker-compose up
For GPU
$ docker-compose --file docker-compose-gpu.yml up
```
