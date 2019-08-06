import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import argparse
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=(160,576)):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))

        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, required=True,help="path to checkpoint model")
parser.add_argument("--out_folder", type=str, required=True,help="path to checkpoint model")

opt = parser.parse_args()

if not os.path.exists(opt.out_folder):
    os.makedirs(opt.out_folder)

dataloader = DataLoader(
    ImageFolder(opt.image_folder, img_size=(160,576)),
    batch_size=1,
    shuffle=False,
    num_workers=4,
)

for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    img = input_imgs.numpy()
    img = (img[0]).transpose((1,2,0))
    img = (img*255).astype(np.uint8)
    cv2.imwrite(opt.out_folder+'output'+str(batch_i)+'.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))