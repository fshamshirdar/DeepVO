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

from deepvonet import DeepVONet
from dataset import VisualOdometryDataLoader

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
K = 100.

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def train_model(model, train_loader, criterion, optimizer, epoch, batch_size, trajectory_length):
    # switch to train mode
    for batch_idx, (images_stacked, odometries_stacked) in enumerate(train_loader):
        if USE_CUDA:
            images_stacked, odometries_stacked = images_stacked.cuda(), odometries_stacked.cuda()
        images_stacked = images_stacked.permute(1, 0, 2, 3, 4)
        images_stacked, odometries_stacked = Variable(images_stacked), Variable(odometries_stacked)

        estimated_odometries = Variable(torch.zeros(odometries_stacked.shape))
        estimated_odometries = estimated_odometries.permute(1, 0, 2)
        if USE_CUDA:
            estimated_odometries = estimated_odometries.cuda()

        model.reset_hidden_states(size=batch_size, zero=True)
        for t in range(trajectory_length):
#            plt.imshow(images_stacked[0][0].data.cpu().numpy().transpose(1, 2, 0))
#            plt.show()

#            f, axarr = plt.subplots(2,2)
#            axarr[0,0].imshow(image1[0].data.cpu().numpy().transpose((1, 2, 0)))
#            axarr[0,1].imshow(image2[0].data.cpu().numpy().transpose((1, 2, 0)))
#            plt.show()

            # compute output
            estimated_odometry = model(images_stacked[t])
            estimated_odometries[t] = estimated_odometry

        estimated_odometries = estimated_odometries.permute(1, 0, 2)
        loss = criterion(estimated_odometries[:,:,:3], odometries_stacked[:,:,:3]) + K * criterion(estimated_odometries[:,:,3:], odometries_stacked[:,:,3:])

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print (epoch, batch_idx, loss.data.cpu()[0])

def train(model, datapath, checkpoint_path, epochs, trajectory_length, args):
    model.train()
    model.training = True

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(VisualOdometryDataLoader(datapath, trajectory_length=trajectory_length, transform=preprocess), batch_size=args.bsize, shuffle=True, drop_last=True, **kwargs)

    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(1, epochs + 1):
        # train for one epoch
        train_model(model, train_loader, criterion, optimizer, epoch, args.bsize, trajectory_length)
#        # evaluate on validation set
#        acc = test(test_loader, tripletnet, criterion, epoch)
#
#        # remember best acc and save checkpoint
#        is_best = acc > best_acc
#        best_acc = max(acc, best_acc)
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_{}.pth".format(epoch)))

def test_model(model, test_loader, batch_size, trajectory_length):
    for batch_idx, (images_stacked, odometries_stacked) in enumerate(test_loader):
        if USE_CUDA:
            images_stacked, odometries_stacked = images_stacked.cuda(), odometries_stacked.cuda()
        images_stacked = images_stacked.permute(1, 0, 2, 3, 4)
        images_stacked, odometries_stacked = Variable(images_stacked), Variable(odometries_stacked)

        estimated_odometries = Variable(torch.zeros(odometries_stacked.shape))
        estimated_odometries = estimated_odometries.permute(1, 0, 2)
        if USE_CUDA:
            estimated_odometries = estimated_odometries.cuda()

        model.reset_hidden_states(size=batch_size, zero=True)
        for t in range(trajectory_length):
            estimated_odometry = model(images_stacked[t])
            estimated_odometries[t] = estimated_odometry

        estimated_odometries = estimated_odometries.permute(1, 0, 2)
        print (estimated_odometries, odometries_stacked)

def test(model, datapath, trajectory_length, validation_steps, preprocess):
    model.eval()
    model.training = False

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_loader = torch.utils.data.DataLoader(VisualOdometryDataLoader(datapath, trajectory_length=trajectory_length, transform=preprocess, test=True), batch_size=1, shuffle=True, **kwargs)
 
    for epoch in range(1, validation_steps+1):
        test_model(model, test_loader, 1, trajectory_length)

    """
    with open(os.path.join(datapath, "index.txt"), 'r') as reader:
        for index in reader:
            index = index.strip()
            images_path = []
            with open(os.path.join(datapath, index, "index.txt"), 'r') as image_reader:
                for image_path in image_reader:
                    images_path.append(image_path.strip())

            model.reset_hidden_states(size=1, zero=True)
            for image_index in range(len(images_path)-1):
                model.reset_hidden_states(size=1, zero=False)
                image1 = Image.open(os.path.join(datapath, index, images_path[image_index])).convert('RGB')
                image2 = Image.open(os.path.join(datapath, index, images_path[image_index+1])).convert('RGB')
                image1_tensor = preprocess(image1)
                image2_tensor = preprocess(image2)

                # plt.figure()
                # plt.imshow(images_stacked.cpu().numpy().transpose((1, 2, 0)))
                # plt.show()

                images_stacked = torch.from_numpy(np.concatenate([image1_tensor, image2_tensor], axis=0))
                images_stacked.unsqueeze_(0)
                images_stacked = Variable(images_stacked).cuda()
                odom = model(images_stacked)
                print (image_index, image_index+1, odom.data.cpu())
                del images_stacked, odom, image1_tensor, image2_tensor
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch on Place Recognition + Visual Odometry')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--datapath', default='datapath', type=str, help='path KITII odometry dataset')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--trajectory_length', default=10, type=int, help='trajectory length')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_iter', default=20000000, type=int, help='train iters each timestep')
    parser.add_argument('--validation_steps', default=100, type=int, help='test iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Checkpoint path')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint')
    args = parser.parse_args()

    normalize = transforms.Normalize(
        #mean=[121.50361069 / 127., 122.37611083 / 127., 121.25987563 / 127.],
        mean=[127. / 255., 127. / 255., 127. / 255.],
        std=[1 / 255., 1 / 255., 1 / 255.]
    )

    preprocess = transforms.Compose([
        transforms.Resize((384, 1280)),
        transforms.CenterCrop((384, 1280)),
        transforms.ToTensor(),
        normalize
    ])

    model = DeepVONet()
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
    if USE_CUDA:
        model.cuda()

    args = parser.parse_args()
    if args.mode == 'train':
        train(model, args.datapath, args.checkpoint_path, args.train_iter, args.trajectory_length, args)
    elif args.mode == 'test':
        test(model, args.datapath, args.trajectory_length, args.validation_steps, preprocess)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
