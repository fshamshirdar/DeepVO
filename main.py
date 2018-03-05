import io
import os
from PIL import Image
import numpy as np
import time
import math
import matplotlib.pyplot as plt

import argparse
import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from placenet import PlaceNet
from odometrynet import OdometryNet
from dataset import VisualOdometryDataLoader

model = PlaceNet()

def train_model(train_loader, odometrynet, criterion, optimizer, epoch):
    # switch to train mode
    odometrynet.train()
    for batch_idx, (image1, image2, odometry) in enumerate(train_loader):
        if torch.cuda.is_available():
            image1, image2, odometry = image1.cuda(), image2.cuda(), odometry.cuda()
        image1, image2, odometry = Variable(image1), Variable(image2), Variable(odometry)

#        f, axarr = plt.subplots(2,2)
#        axarr[0,0].imshow(image1[0].data.cpu().numpy().transpose((1, 2, 0)))
#        axarr[0,1].imshow(image2[0].data.cpu().numpy().transpose((1, 2, 0)))
#        plt.show()

        # compute output
        # estimated_odometry = odometrynet(image1, image2)
        # loss = criterion(estimated_odometry, odometry)

        estimated_yaw = odometrynet(image1, image2)
        print (odometry - estimated_yaw)

        loss = criterion(estimated_yaw, odometry)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print (loss)

def train(odometrynet, datapath, checkpoint_path, epochs, preprocess, args):
#    model.train()
    model.training = False

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(VisualOdometryDataLoader(datapath, transform=preprocess), batch_size=args.bsize, shuffle=True, **kwargs)

    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(odometrynet.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, epochs + 1):
        # train for one epoch
        train_model(train_loader, odometrynet, criterion, optimizer, epoch)
#        # evaluate on validation set
#        acc = test(test_loader, tripletnet, criterion, epoch)
#
#        # remember best acc and save checkpoint
#        is_best = acc > best_acc
#        best_acc = max(acc, best_acc)
        state = {
            'epoch': epoch + 1,
            'state_dict': odometrynet.state_dict(),
        }
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_{}.pth".format(epoch)))

def test_model(test_loader, odometrynet):
    # switch to test mode
    odometrynet.eval()
    for batch_idx, (image1, image2, odometry) in enumerate(train_loader):
        if torch.cuda.is_available():
            image1, image2, odometry = image1.cuda(), image2.cuda(), odometry.cuda()
        image1, image2, odometry = Variable(image1), Variable(image2), Variable(odometry)

        # compute output
        # estimated_odometry = odometrynet(image1, image2)
        # loss = criterion(estimated_odometry, odometry)

        estimated_yaw = odometrynet(image1, image2)
        print (odometry - estimated_yaw)

def test(odometrynet, testpath, validation_steps, preprocess):
    model.eval()
    model.training = False
    odometrynet.eval()
    odometrynet.training = False

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_loader = torch.utils.data.DataLoader(VisualOdometryDataLoader(datapath, transform=preprocess, test=True), batch_size=args.bsize, shuffle=True, **kwargs)

    for epoch in range(1, validation_steps+1):
        test_model(test_loader, odometrynet)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch on Place Recognition + Visual Odometry')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--datapath', default='datapath', type=str, help='path KITII odometry dataset')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_iter', default=20000000, type=int, help='train iters each timestep')
    parser.add_argument('--validation_steps', default=100, type=int, help='test iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Checkpoint path')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint')
    parser.add_argument('--place_checkpoint', default=None, type=str, help='Place Checkpoint')
    args = parser.parse_args()

    normalize = transforms.Normalize(
        #mean=[121.50361069 / 127., 122.37611083 / 127., 121.25987563 / 127.],
        mean=[1., 1., 1.],
        std=[1 / 127., 1 / 127., 1 / 127.]
    )

    preprocess = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        normalize
    ])

    place_checkpoint = torch.load(args.place_checkpoint)
    model.load_state_dict(place_checkpoint['state_dict'])
    if torch.cuda.is_available():
        model.cuda()

    odometrynet = OdometryNet(model)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        odometrynet.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available():
        odometrynet.cuda()

    args = parser.parse_args()
    if args.mode == 'train':
        train(odometrynet, args.datapath, args.checkpoint_path, args.train_iter, preprocess, args)
    elif args.mode == 'test':
        test(odometrynet, args.datapath, args.validation_steps, preprocess)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
