import os
from glob import glob
import re
import torch.nn as nn
from torch.autograd import Variable
import torch
import cv2
import json
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import scipy.misc

np.random.seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(img, mean, std):
    img = img/255.0
    img[0] = (img[0] - mean[0]) / std[0]
    img[1] = (img[1] - mean[1]) / std[1]
    img[2] = (img[2] - mean[2]) / std[2]
    img = np.clip(img, 0.0, 1.0)

    return img

def denormalize(img, mean, std):
    img[0] = (img[0] * std[0]) + mean[0]
    img[1] = (img[1] * std[1]) + mean[1]
    img[2] = (img[2] * std[2]) + mean[2]
    img = img * 255

    img = np.clip(img, 0, 255)
    return img

def get_label_paths(label_path):
    label_paths = {re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                   for path in glob(os.path.join(label_path, '*_road_*.png'))}

    return label_paths

def get_test_paths(test_path):
    test_paths = [os.path.basename(path)
                      for path in glob(os.path.join(test_path, '*.png'))]

    return test_paths

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

def gen_test_output(n_class, testloader, model, test_folder):
    model.eval();
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images = data['image']
            images = images.float()
            images = Variable(images.to(device))

            output = model(images)
            output = torch.sigmoid(output)
            output = output.cpu()
            N, c, h, w = output.shape
            pred = np.squeeze(output.detach().cpu().numpy(), axis=0)

            pred = pred.transpose((1, 2, 0))
            pred = pred.argmax(axis=2)
            segmentation = (pred > 0.5) # get pred map
            segmentation = segmentation.reshape(*segmentation.shape, 1)


            image_file = data['name'][0]
            raw_image = image = scipy.misc.imresize(scipy.misc.imread(image_file), (256,256))
            mask = np.dot(segmentation, np.array([[0, 255, 255, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(raw_image)
            street_im.paste(mask, box=None, mask=mask)
            test_paths = get_test_paths(test_folder)
            output = np.array(street_im)
            yield test_paths[i], output

def save_inference_samples(output_dir, testloader, model, test_folder):
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(2, testloader, model, test_folder)
    for name, image in image_outputs:
        plt.imsave(os.path.join(output_dir, name), image)

