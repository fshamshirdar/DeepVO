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

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def train_model(model, train_loader, criterion, optimizer, epoch, trajectory_length):
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

        model.reset_hidden_states(zero=True)
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
        loss = criterion(estimated_odometries, odometries_stacked)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print (loss)

def train(model, datapath, checkpoint_path, epochs, trajectory_length, args):
#    model.train()
    model.training = False

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(VisualOdometryDataLoader(datapath, trajectory_length=trajectory_length, transform=preprocess), batch_size=args.bsize, shuffle=True, **kwargs)

    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, epochs + 1):
        # train for one epoch
        train_model(model, train_loader, criterion, optimizer, epoch, trajectory_length)
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

def test(model, datapath, preprocess):
    model.eval()
    model.training = False

    with open(os.path.join(datapath, "index.txt"), 'r') as reader:
        import torch.nn.functional as F
        reps = []
        for index in reader:
            index = index.strip()
            with open(os.path.join(datapath, index, "index.txt"), 'r') as image_reader:
                for image_path in image_reader:
                    print (image_path)
                    image_path = image_path.strip()
                    image = Image.open(os.path.join(datapath, index, image_path)).convert('RGB')
                    image_tensor = preprocess(image)

#                    plt.figure()
#                    plt.imshow(image_tensor.cpu().numpy().transpose((1, 2, 0)))
#                    plt.show()

                    image_tensor.unsqueeze_(0)
                    image_variable = Variable(image_tensor).cuda()
                    features = model(image_variable)
                    reps.append(features.data.cpu())

                for i in range(len(reps)):
                    print ("\n\n")
                    for j in range(len(reps)):
                        # d = np.asarray(reps[j] - reps[i])
                        # similarity = np.linalg.norm(d)
                        # print (i, j, similarity)

                        similarity = F.pairwise_distance(reps[j], reps[i], 2)
                        print (i, j, similarity[0][0])

                        # similarity = F.cosine_similarity(reps[j], reps[i])
                        # print (i, j, similarity[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch on Place Recognition + Visual Odometry')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--datapath', default='datapath', type=str, help='path KITII odometry dataset')
    parser.add_argument('--bsize', default=32, type=int, help='minibatch size')
    parser.add_argument('--trajectory_length', default=10, type=int, help='trajectory length')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_iter', default=20000000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Checkpoint path')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint')
    args = parser.parse_args()

    normalize = transforms.Normalize(
        #mean=[121.50361069 / 127., 122.37611083 / 127., 121.25987563 / 127.],
        mean=[1., 1., 1.],
        std=[1 / 127., 1 / 127., 1 / 127.]
    )

    preprocess = transforms.Compose([
        transforms.Resize((384, 1280)),
        transforms.CenterCrop((384, 1280)),
        transforms.ToTensor(),
        # normalize
    ])

    model = DeepVONet(args.bsize, USE_CUDA)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
    if USE_CUDA:
        model.cuda()

    args = parser.parse_args()
    if args.mode == 'train':
        train(model, args.datapath, args.checkpoint_path, args.train_iter, args.trajectory_length, args)
    elif args.mode == 'test':
        test(model, args.datapath, preprocess)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
