import argparse
from random import randint
from time import sleep

import torch
import torch.distributed as dist


def foo(rank, world_size, steps):
    for step in range(1, steps + 1):
        # get random int
        value = randint(0, 10)

        # group all ranks
        ranks = list(range(world_size))
        group = dist.new_group(ranks=ranks)

        # compute reduced sum
        tensor = torch.IntTensor([value])
        dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)

        print('rank: {}, step: {}, value: {}, reduced sum: {}.'.format(
            rank, step, value, float(tensor)))

        sleep(1)


def init_process(backend, init_method, rank, world_size):
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backend',
        type=str,
        default='tcp',
        help='Name of the backend to use.')
    parser.add_argument(
        '--init-method',
        '-i',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument(
        '--rank', '-r', type=int, help='Rank of the current process.')
    parser.add_argument(
        '--world-size',
        '-s',
        type=int,
        help='Number of processes participating in the job.')
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    print(args)

    init_process(args.backend, args.init_method, args.rank, args.world_size)
    foo(args.rank, args.world_size, args.steps)


if __name__ == '__main__':
    main()
