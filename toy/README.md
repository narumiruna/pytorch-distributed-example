# Toy Example

Rank 0
```
$ python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=1234 \
    main.py
```

Rank 2
```
$ python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=1234 \
    main.py
```
