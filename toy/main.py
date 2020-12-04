import argparse
from random import randint
from time import sleep

import torch
import torch.distributed as dist


def run(local_rank, steps):
    for step in range(1, steps + 1):
        # get random int
        value = randint(0, 10)

        # group all ranks
        ranks = list(range(dist.get_world_size()))
        group = dist.new_group(ranks=ranks)

        # compute reduced sum
        tensor = torch.tensor(value, dtype=torch.int)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

        print(
            f'node_rank: {dist.get_rank()}'
            f', local_rank: {local_rank}'
            f', step: {step}'
            f', value: {value}'
            f', reduced sum: {tensor.item()}'
            )

        sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    print(args)

    dist.init_process_group(backend=args.backend)

    run(args.local_rank, args.steps)


if __name__ == '__main__':
    main()
