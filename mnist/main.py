import argparse

import torch
from torch import distributed, nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms

import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(32, 64, 5, 2, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(),
        )

        self.linear = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10)
        )

    def forward(self, input_):
        output = self.conv(input_)
        output = output.view(-1, 64 * 4 * 4)
        output = self.linear(output)
        return output


def get_dataloader(root, batch_size, shuffle=True, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066047740239478,), (0.3081078087569972,))
    ])

    train_loader = data.DataLoader(datasets.MNIST(root,
                                                  train=True,
                                                  transform=transform,
                                                  download=True),
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers)

    test_loader = data.DataLoader(datasets.MNIST(root,
                                                 train=False,
                                                 transform=transform,
                                                 download=True),
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)

    return train_loader, test_loader


def evaluate(args, net, test_dataloader):
    net.eval()

    correct = 0
    for test_index, (test_x, test_y) in enumerate(test_dataloader):
        test_x = Variable(test_x, volatile=True)

        if args.cuda:
            test_x = test_x.cuda()
            test_y = test_y.cuda()

        _, max_indices = net(test_x).data.max(1)
        correct += int((max_indices == test_y).sum())

    accuracy = correct / len(test_dataloader.dataset)

    return accuracy


def average_gradients(net):
    world_size = distributed.get_world_size()

    for p in net.parameters():
        group = distributed.new_group(ranks=list(range(world_size)))
        distributed.all_reduce(p.grad.data,
                               op=distributed.reduce_op.SUM,
                               group=group)
        p.grad.data /= float(world_size)


def train(args, epoch, net, optimizer, train_loader, test_loader):
    for train_index, (train_x, train_y) in enumerate(train_loader):
        net.train()

        train_x = Variable(train_x)
        train_y = Variable(train_y)

        if args.cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        pred_y = net(train_x)
        loss = F.cross_entropy(pred_y, train_y)

        loss.backward()
        # average the gradients
        average_gradients(net)

        optimizer.step()
        optimizer.zero_grad()

        if train_index % args.log_interval == 0:

            accuracy = evaluate(args, net, test_loader)

            print('Train epoch: {}, batch index: {}, loss: {}, accuracy: {}.'.format(epoch,
                                                                                     train_index,
                                                                                     float(loss.data),
                                                                                     accuracy))


def solve(args):
    net = Net()
    if args.cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    train_loader, test_loader = get_dataloader(args.data_dir,
                                               args.batch_size,
                                               num_workers=args.num_workers)

    for epoch in range(args.epochs):
        train(args, epoch, net, optimizer, train_loader, test_loader)


def init_process(args):
    distributed.init_process_group(backend=args.backend,
                                   init_method='{}://{}:{}'.format(
                                       args.backend, args.ip, args.port),
                                   rank=args.rank,
                                   world_size=args.world_size)

    solve(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='20000')
    parser.add_argument('--rank', '-r', type=int)
    parser.add_argument('--world-size', '-s', type=int)
    parser.add_argument('--backend', type=str, default='tcp')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=1000)
    args = parser.parse_args()
    print(args)

    init_process(args)


def test_mnist():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=1000)
    args = parser.parse_args()
    print(args)

    solve(args)


if __name__ == '__main__':
    main()
    # test_mnist()
