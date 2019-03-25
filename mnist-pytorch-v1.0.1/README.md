# MNIST Example for PyTorch v1.0.1

Rank 0
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 2
```

Rank 2
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 2
```

## Use specific root directory for running example on single machine.

Rank 0
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 2 --root data0
```

Rank 2
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 2 --root data1
```

## Run in docker

Install [docker](https://docs.docker.com/install/) and [docker-compose](https://docs.docker.com/compose/install/)

```
$ docker build --file Dockerfile --tag pytorch-distributed-example .
$ docker-compose up
```

