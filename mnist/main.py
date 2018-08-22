import argparse

import torch
import torch.nn.functional as F
from torch import distributed, nn
from torch.utils import data
from torchvision import datasets, transforms


class Trainer(object):

    def __init__(self, net, optimizer, train_loader, test_loader, device):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def train(self):
        train_loss = AverageMeter()
        train_acc = AccuracyMeter()

        self.net.train()

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            out = self.net(x)
            loss = F.cross_entropy(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            # average the gradients
            self.average_gradients()
            self.optimizer.step()

            y_pred = out.data.argmax(dim=1)
            correct = y_pred.eq(y.data).sum().item()

            train_loss.update(loss.data.item(), x.size(0))
            train_acc.update(correct, x.size(0))

        return train_loss.average, train_acc.accuracy

    def evaluate(self):
        test_loss = AverageMeter()
        test_acc = AccuracyMeter()

        self.net.eval()

        for x, y in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            out = self.net(x)
            loss = F.cross_entropy(out, y)
            y_pred = out.data.argmax(dim=1)
            correct = y_pred.eq(y.data).sum().item()

            test_loss.update(loss.data.item(), x.size(0))
            test_acc.update(correct, x.size(0))

        return test_loss.average, test_acc.accuracy

    def average_gradients(self):
        world_size = distributed.get_world_size()

        for p in self.net.parameters():
            group = distributed.new_group(ranks=list(range(world_size)))

            tensor = p.grad.data.cpu()

            distributed.all_reduce(
                tensor, op=distributed.reduce_op.SUM, group=group)

            tensor /= float(world_size)

            p.grad.data = tensor.to(self.device)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out


def get_dataloader(root, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066047740239478,), (0.3081078087569972,))
    ])

    train_loader = data.DataLoader(
        datasets.MNIST(root, train=True, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=True)

    test_loader = data.DataLoader(
        datasets.MNIST(root, train=False, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=False)

    return train_loader, test_loader


class AverageMeter(object):

    def __init__(self):
        self.sum = 0
        self.count = 0
        self.average = None

    def update(self, value, number):
        self.sum += value * number
        self.count += number
        self.average = self.sum / self.count


class AccuracyMeter(object):

    def __init__(self):
        self.correct = 0
        self.count = 0
        self.accuracy = None

    def update(self, correct, number):
        self.correct += correct
        self.count += number
        self.accuracy = self.correct / self.count


def solve(args):
    device = torch.device('cuda' if args.cuda else 'cpu')

    net = Net().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    train_loader, test_loader = get_dataloader(args.root, args.batch_size)

    trainer = Trainer(net, optimizer, train_loader, test_loader, device)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = trainer.train()
        with torch.no_grad():
            test_loss, test_acc = trainer.evaluate()

        print(
            'Train epoch: {}/{},'.format(epoch, args.epochs),
            'train loss: {:.6f}, train acc: {:.6f}, test loss: {:.6f}, test acc: {:.6f}.'.
            format(train_loss, train_acc, test_loss, test_acc))


def init_process(args):
    distributed.init_process_group(
        backend=args.backend,
        init_method='{}://{}:{}'.format(args.backend, args.ip, args.port),
        rank=args.rank,
        world_size=args.world_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='20000')
    parser.add_argument('--rank', '-r', type=int)
    parser.add_argument('--world-size', '-s', type=int)
    parser.add_argument('--backend', type=str, default='tcp')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    init_process(args)
    solve(args)


if __name__ == '__main__':
    main()
