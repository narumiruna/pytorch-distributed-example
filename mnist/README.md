# MNIST Example



Machine 1 with ip 10.1.14.2 port 23000
```
$ python3 main.py --ip 10.1.14.132 --port 23000 --rank 0 --world-size 3 --cuda
```

Machine 2
```
$ python3 main.py --ip 10.1.14.132 --port 23000 --rank 1 --world-size 3 --cuda
```

Machine 3
```
$ python3 main.py --ip 10.1.14.132 --port 23000 --rank 2 --world-size 3 --cuda
```
