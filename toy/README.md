# Toy Example

## In single machine

Terminal 1
```
$ python main.py --rank 0 --size 2
```

Terminal 2
```
$ python main.py --rank 1 --size 2
```

## Use two machines

Machine 1 with ip 10.1.14.2
```
$ python main.py --rank 0 --size 2 --ip 10.1.14.2 --port 22000
```

Machine 2
```
$ python main.py --rank 1 --size 2 --ip 10.1.14.2 --port 22000
```
