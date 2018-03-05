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

def train(odometrynet, datapath, checkpoint_path, epochs, args):
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

def test(odometrynet, datapath, preprocess):
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
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_iter', default=20000000, type=int, help='train iters each timestep')
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
        train(odometrynet, args.datapath, args.checkpoint_path, args.train_iter, args)
    elif args.mode == 'test':
        test(odometrynet, args.datapath, preprocess)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
