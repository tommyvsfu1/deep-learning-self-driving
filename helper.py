import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
from glob import glob
from glob import glob1
from urllib.request import urlretrieve
from tqdm import tqdm
import matplotlib.pyplot as plt

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def get_csv_file(data_folder):
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): os.path.basename(path)
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}


    import csv

    # 開啟輸出的 CSV 檔案
    with open('train.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        for image, label in label_paths.items():
            writer.writerow([image, label])

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): os.path.basename(path)
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}


        import csv

        # 開啟輸出的 CSV 檔案
        with open('output.csv', 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            for image, label in label_paths.items():
                writer.writerow([image, label])

        print("after")
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            # for batch_i in range(0,1):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2) # get backgroud feature map
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)  # expand dim

                # get ground truth
                # tricks: since we only want 2 class (background and road)
                # so use invert(background), we can get road feature map
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2) 
                


                images.append(image)
                gt_images.append(gt_image)

            
            
            
            yield np.array(images), np.array(gt_images)
    return get_batches_fn





def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to disk
    print('Training Finished!')
    print('Saving test images to: {}, please wait...'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    print('All augmented images are saved!')



def pred_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, print_speed=False):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Predicting images...')
    # start epoch training timer

    image_outputs = gen_output(
        sess, logits, keep_prob, input_image, data_dir, image_shape)

    counter = 0
    for name, image, speed_ in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        if print_speed is True:
            counter+=1
            print("Processing file: {0:05d},\tSpeed: {1:.2f} fps".format(counter, speed_))

        # sum_time += laptime

    # pngCounter = len(glob1(data_dir,'*.png'))

    print('All augmented images are saved to: {}.'.format(output_dir))




# image_shape = (160, 576)
# data_dir = './data'
# get_batches_fn = gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
# for image, label in get_batches_fn(1):
#     pass
#     # print("type of image:", image.shape)
#     # print("type of label:", label.shape)

data_dir = './data'
train_dir = os.path.join(data_dir, 'data_road/training')
get_csv_file(train_dir)

