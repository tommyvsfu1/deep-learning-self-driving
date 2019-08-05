from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.misc import imread
from PIL import Image
import cv2
import sys
import random
random.seed(11037)
np.set_printoptions(threshold=sys.maxsize)
class KittiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, crop=True, flip_rate=0.5):
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
        self.flip_rate = flip_rate
        self.crop = crop
        self.new_h = 256
        self.new_w = 256
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        
        image_prefix = self.root_dir + "image_2/"
        gt_prefix = self.root_dir + "gt_image_2/"
        image_name = str(self.frame.iloc[idx, 0]) 
        gt_name = str(self.frame.iloc[idx, 1])



        background_color = np.array([255, 0, 0])
        gt_name = os.path.join(gt_prefix,gt_name)
        # gt_image = scipy.misc.imread(gt_name)
        gt_image = imread(gt_name)
        gt_image = cv2.resize(gt_image, (576,160))
        gt_bg = np.all(gt_image == background_color, axis=2) # get backgroud feature map
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)  # expand dim
        # get ground truth
        # tricks: since we only want 2 class (background and road)
        # so use invert(background), we can get road feature map
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2) 
        # gt_image = Image.fromarray(gt_image)
        gt_map = np.invert(gt_bg) 



        # if self.crop:
        #     h, w, _ = image.shape
        #     top   = random.randint(0, h - self.new_h)
        #     left  = random.randint(0, w - self.new_w)
        #     image   = image[top:top + self.new_h, left:left + self.new_w]
        #     gt_image = gt_image[top:top + self.new_h, left:left + self.new_w]
        #     gt_map = gt_map[top:top + self.new_h, left:left + self.new_w]

        # if random.random() < self.flip_rate:
        #     image   = np.fliplr(image)
        #     gt_image = np.fliplr(gt_image)
        #     gt_map = np.fliplr(gt_map)

        gt_image = gt_image.transpose((2,0,1))
        gt_image = gt_image.astype("float")

        img_name = os.path.join(image_prefix,image_name)
        image = cv2.imread(img_name) # read as np.array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # HSV augmentation
        image = perform_augmentation(image)
        image = Image.fromarray(image) # convert to PIL image(Pytorch default image datatype)
        sample = {'image': image.copy(), 'label': torch.from_numpy(gt_image.copy()), 'gt_map':torch.from_numpy(gt_map.copy())}

        if self.transform:
            sample['image'] = self.transform['train_x'](sample['image'])

        return sample

def load_Kitti(batch_size, split=True):

    data_transforms = {
        'train_x': transforms.Compose([
            transforms.Resize((160,576)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    dataset = KittiDataset('data/train.csv','data/data_road/training/',transform=data_transforms)

    if split:
        train_dataloader, val_dataloader = datasetSplit(dataset=dataset, train_batch_size=batch_size)
    else :
        dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    return train_dataloader, val_dataloader

def datasetSplit(dataset, train_batch_size):
    validation_split = .1
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size,
                                                shuffle=False,num_workers=4,sampler=train_sampler)
                
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                shuffle=False,num_workers=4,sampler=valid_sampler)
    return train_dataloader, val_dataloader

class KittiTestDataset(Dataset):
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
        image_name = str(self.frame.iloc[idx, 0]) 

        img_name = os.path.join(image_prefix,image_name)
        raw_image = imread(img_name) # read as np.array
        image = Image.fromarray(raw_image) # convert to PIL image(Pytorch default image datatype)
        if self.transform:
            image = self.transform(image)
        sample = {'image':image, 'name':img_name}
        return sample

def load_Kitti_test(batch_size):

    data_transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset = KittiTestDataset('data/test.csv','data/data_road/testing/',transform=data_transforms)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

    return dataloader



def perform_augmentation(batch_x):
    """
    Perform basic data augmentation on image batches.
    
    Parameters
    ----------
    batch_x: ndarray of shape (b, h, w, c)
        Batch of images in RGB format, values in [0, 255]
    batch_y: ndarray of shape (b, h, w, c)
        Batch of ground truth with road segmentation
        
    Returns
    -------
    batch_x_aug, batch_y_aug: two ndarray of shape (b, h, w, c)
        Augmented batches
    """
    def mirror(x):
        return x[:, ::-1, :]

    def augment_in_hsv_space(x_hsv):
        x_hsv = np.float32(cv2.cvtColor(x_hsv, cv2.COLOR_RGB2HSV))
        x_hsv[:, :, 0] = x_hsv[:, :, 0] * random.uniform(0.9, 1.1)   # change hue
        x_hsv[:, :, 1] = x_hsv[:, :, 1] * random.uniform(0.5, 2.0)   # change saturation
        x_hsv[:, :, 2] = x_hsv[:, :, 2] * random.uniform(0.5, 2.0)   # change brightness
        x_hsv = np.uint8(np.clip(x_hsv, 0, 255))
        return cv2.cvtColor(x_hsv, cv2.COLOR_HSV2RGB)

    batch_x_aug = np.copy(batch_x)

    # Random change in image values (hue, saturation, brightness)
    batch_x_aug = augment_in_hsv_space(batch_x_aug)

    return batch_x_aug
