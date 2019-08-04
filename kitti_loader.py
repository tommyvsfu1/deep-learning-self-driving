from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms, utils
from scipy.misc import imread
from PIL import Image
import cv2

class KittiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        
        image_prefix = self.root_dir + "image_2/"
        gt_prefix = self.root_dir + "gt_image_2/"
        image_name = str(self.frame.iloc[idx, 0]) 
        gt_name = str(self.frame.iloc[idx, 1])

        img_name = os.path.join(image_prefix,image_name)
        image = imread(img_name) # read as np.array
        image = Image.fromarray(image) # convert to PIL image(Pytorch default image datatype)

        background_color = np.array([255, 0, 0])
        gt_name = os.path.join(gt_prefix,gt_name)
        # gt_image = scipy.misc.imread(gt_name)
        gt_image = imread(gt_name)
        gt_image = cv2.resize(gt_image, (256,256))
        gt_bg = np.all(gt_image == background_color, axis=2) # get backgroud feature map
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)  # expand dim
        # get ground truth
        # tricks: since we only want 2 class (background and road)
        # so use invert(background), we can get road feature map
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2) 
        gt_image = gt_image.transpose((2,0,1))
        gt_image = gt_image.astype("float")

        # gt_image = Image.fromarray(gt_image)
        sample = {'image': image, 'label': torch.from_numpy(gt_image)}

        if self.transform:
            sample['image'] = self.transform['train_x'](sample['image'])
        return sample

def load_Kitti(batch_size):

    data_transforms = {
        'train_x': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    dataset = KittiDataset('data/train.csv','data/data_road/training/',transform=data_transforms)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

    return dataloader

