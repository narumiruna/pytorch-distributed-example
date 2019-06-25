import argparse
from random import randint
from time import sleep

import torch
import torch.distributed as dist


def run(world_size, rank, steps):
    for step in range(1, steps + 1):
        # get random int
        value = randint(0, 10)

        # group all ranks
        ranks = list(range(world_size))
        group = dist.new_group(ranks=ranks)

        # compute reduced sum
        tensor = torch.tensor(value, dtype=torch.int)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

        print('rank: {}, step: {}, value: {}, reduced sum: {}.'.format(rank, step, value, tensor.item()))

        sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument('-s', '--world-size', type=int, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    print(args)

    dist.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        world_size=args.world_size,
        rank=args.rank,
    )

    run(args.world_size, args.rank, args.steps)


if __name__ == '__main__':
    main()
