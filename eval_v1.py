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
# from kitti_loader import load_Kitti, load_Kitti_test, load_Kitti_raw_test
from utils import save_inference_samples, save_model, load_model
from scipy.misc import imread
import sys
# np.set_printoptions(threshold=sys.maxsize)
np.random.seed(1234)


# parser = argparse.ArgumentParser()

# parser.add_argument('--input_dir', type=str, required=True)
# parser.add_argument('--test_img_size', type=str, required=True,
#                     help='resize size before into network')
# parser.add_argument('--output_dir', type=str, required=True,
#                     help='output directory for test inference')


# parser.add_argument('--model', type=str, default='vgg19',
#                     help='model architecture to be used for FCN')
# parser.add_argument('--epochs', type=int, default=150,
#                     help='num of training epochs')
# parser.add_argument('--n_class', type=int, default=2,
#                     help='number of label classes')
# parser.add_argument('--batch_size', type=int, default=16,
#                     help='training batch size')
# parser.add_argument('--lr', type=float, default=1e-3,
#                     help='learning rate')
# parser.add_argument('--momentum', type=float, default=0.9,
#                     help='momentum for SGD')
# parser.add_argument('--weight_decay', type=float, default=5e-4,
#                     help='weight decay for L2 penalty')
# args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("using device:", device)

# # create dir for score
# score_dir = os.path.join("./scores")
# if not os.path.exists(score_dir):
#     os.makedirs(score_dir)
# IU_scores    = np.zeros((args.epochs, args.n_class))
# pixel_scores = np.zeros(args.epochs)

# def train(n_epoch, trainloader, val_loader):
#     vgg_model = VGGNet(requires_grad=True, remove_fc=True)
#     model = FCNs(pretrained_net=vgg_model, n_class=2).to(device)
    
#     #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
#     #                            momentum=args.momentum, weight_decay=args.weight_decay)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     criterion = nn.BCELoss()

#     best_road_iou =  float('-inf')
    
#     for epoch in range(n_epoch):
#         running_loss = 0.0
#         # training
#         for i, data in enumerate(trainloader):
#             sample = data
#             images = sample['image']
#             images = images.float()
#             labels = sample['label']
#             labels = labels.float()
#             images = Variable(images.cuda())
#             labels = Variable(labels.cuda(), requires_grad=False)

#             optimizer.zero_grad()
#             output = model(images)
#             output = torch.sigmoid(output)
#             loss = criterion(output, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             if i % 10 == 9:    # print every 10 mini-batches
#                 print('Epoch: %d, Loss: %.4f' %
#                       (epoch + 1, running_loss / 10))
#                 running_loss = 0.0
#         # validation
#         ious = val(model, val_loader, epoch)
#         if ious[1] > best_road_iou: # save model for best road iou
#             best_road_iou = ious[1]
#             save_model(model,name='best_seg.cpt')
#             print("save best model")

#     return model




# def val(fcn_model, val_loader, epoch):
#     fcn_model.eval()
#     total_ious = []
#     pixel_accs = []
#     for iter, batch in enumerate(val_loader):
#         if torch.cuda.is_available():
#             inputs = Variable(batch['image'].cuda())
#         else:
#             inputs = Variable(batch['image'])

#         output = fcn_model(inputs)
#         output = output.data.cpu().numpy()
#         N, _, h, w = output.shape
#         pred = output.transpose(0, 2, 3, 1).reshape(-1, args.n_class).argmax(axis=1).reshape(N, h, w)
    
#         target = batch['gt_map'].cpu().numpy().reshape(N, h, w) # batch['l'] is gt

#         for p, t in zip(pred, target):
#             total_ious.append(iou(p, t))
#             pixel_accs.append(pixel_acc(p, t))

#     # Calculate average IoU
#     total_ious = np.array(total_ious).T  # n_class * val_len
#     ious = np.nanmean(total_ious, axis=1)
#     pixel_accs = np.array(pixel_accs).mean()
#     print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
#     IU_scores[epoch] = ious
#     np.save(os.path.join(score_dir, "meanIU"), IU_scores)
#     pixel_scores[epoch] = pixel_accs
#     np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)
#     return ious

# # borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# # Calculates class intersections over unions
# def iou(pred, target):
#     ious = []
#     for cls in range(args.n_class):
#         pred_inds = pred == cls
#         target_inds = target == cls
#         intersection = pred_inds[target_inds].sum()
#         union = pred_inds.sum() + target_inds.sum() - intersection
#         if union == 0:
#             ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
#         else:
#             ious.append(float(intersection) / max(union, 1))
#         # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
#     return ious

# def pixel_acc(pred, target):
#     correct = (pred == target).sum()
#     total   = (target == target).sum()
#     return correct / total

# def run():
#     img_size = (int(args.train_img_size.split(',')[0]),int(args.train_img_size.split(',')[1]))
#     kitti_train_loader, kitti_val_loader = load_Kitti(args.input_dir, img_size, args.batch_size)
#     print("Training model..")
#     model = train(args.epochs, kitti_train_loader, kitti_val_loader)
#     print("Completed training!")
#     save_model(model)



# def raw_testing():
#     vgg_model = VGGNet(requires_grad=True, remove_fc=True)
#     model = FCNs(pretrained_net=vgg_model, n_class=2).to(device)
#     model = load_model(model)
#     test_folder = args.input_dir
#     test_data_folder = test_folder + 'data/'
#     print("test_folder", test_folder)
#     img_size = (int(args.test_img_size.split(',')[0]),int(args.test_img_size.split(',')[1]))
#     kitti_raw_test_loader = load_Kitti_raw_test(load_path=test_folder,img_size=img_size,batch_size=1)
#     print("Starting inference...")
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#     save_inference_samples(args.output_dir, img_size, kitti_raw_test_loader,
#                             model, test_data_folder)
#     print("Inference completed!")

# if __name__ == "__main__":
#     # # test 'raw' video data
#     raw_testing()