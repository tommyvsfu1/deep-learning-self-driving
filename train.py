# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
import numpy as np
import os
import argparse
from kitti_loader import load_Kitti, load_Kitti_test
from utils import save_inference_samples
np.random.seed(1234)


parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, required=True,
                    help='output directory for test inference')
parser.add_argument('--model', type=str, default='vgg19',
                    help='model architecture to be used for FCN')
parser.add_argument('--epochs', type=int, default=100,
                    help='num of training epochs')
parser.add_argument('--n_class', type=int, default=2,
                    help='number of label classes')
parser.add_argument('--batch_size', type=int, default=16,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay for L2 penalty')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

def train(n_epoch, trainloader):
    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    model = FCNs(pretrained_net=vgg_model, n_class=2).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()
    """
    for epoch in range(n_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            sample = data
            images = sample['image']
            images = images.float()
            labels = sample['label']
            labels = labels.float()
            images = Variable(images.cuda())
            labels = Variable(labels.cuda(), requires_grad=False)

            optimizer.zero_grad()
            output = model(images)
            output = torch.sigmoid(output)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('Epoch: %d, Loss: %.4f' %
                      (epoch + 1, running_loss / 10))
                running_loss = 0.0
    """
    return model

def main():
    kitti_train_loader = load_Kitti(args.batch_size)
    kitti_test_loader = load_Kitti_test(batch_size=1)
    print("Training model..")
    model = train(args.epochs, kitti_train_loader)
    print("Completed training!")
    print("Starting inference...")
    test_folder = "data/data_road/testing/image_2/"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_inference_samples(args.output_dir, kitti_test_loader,
                            model, test_folder)
    print("Inference completed!")
def testing():
    pass

if __name__ == "__main__":
    main()