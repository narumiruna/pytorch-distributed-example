# MNIST Example

Rank 0
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 3
```

Rank 2
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 3
```

Rank 3
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 2 --world-size 3
```

Use specific root directory for running example on single machine.

Rank 0
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 3 --root data0
```

Rank 2
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 3 --root data1
```

Rank 3
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 2 --world-size 3 --root data2
```
