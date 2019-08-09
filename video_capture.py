import cv2
import numpy as np
import glob
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--fps', type=str, required=True,
                    help='output directory for test inference')
parser.add_argument('--output_dir', type=str, required=True,
                    help='output directory for test inference')

args = parser.parse_args()
def vis_segmentation_stream(image, index):
    """Visualizes segmentation overlay view and stream it with IPython display."""
    plt.figure(figsize=(12, 7))

    #seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(image)
    #plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay | frame #%d'%index)
    plt.grid('off')
    plt.tight_layout()

    # Show visualization in a streaming fashion.
    f = BytesIO()
    plt.savefig(f, format='jpeg')
    IPython.display.display(IPython.display.Image(data=f.getvalue()))
    f.close()
    plt.close()


def run_visualization_video(frame, index):
    """Inferences DeepLab model on a video file and stream the visualization."""
    original_im = Image.fromarray(frame[..., ::-1])
    #seg_map = MODEL.run(original_im)
    vis_segmentation_stream(original_im, index)


def run():
    SAMPLE_VIDEO = 'mit_driveseg_sample.mp4'
    if not os.path.isfile(SAMPLE_VIDEO): 
        print('downloading the sample video...')
        SAMPLE_VIDEO = urllib.request.urlretrieve('https://github.com/lexfridman/mit-deep-learning/raw/master/tutorial_driving_scene_segmentation/mit_driveseg_sample.mp4')[0]
    print('running deeplab on the sample video...')

    video = cv.VideoCapture(SAMPLE_VIDEO)
    # num_frames = 598  # uncomment to use the full sample video
    num_frames = 30

    try:
        for i in range(600):
            _, frame = video.read()
            if not _: break
            run_visualization_video(frame, i)
            IPython.display.clear_output(wait=True)
    except KeyboardInterrupt:
        plt.close()
        print("Stream stopped.")

def read_time_stamp(file_name):
    file = open(file_name, "r") 
    time = []
    avg_fps_factor_list = []
    for line in file: 
        stamp = line.split()[1].split(':')
        time.append(stamp)
        if (len(time) > 1):
            t2 = time[-1][2]
            t1 = time[-2][2]
            avg_fps_factor_list.append(float(t2) - float(t1))
    avg_fps_factor = np.mean(avg_fps_factor_list)
    fps = int(1/avg_fps_factor)
    print("fps", fps)
    return fps

def output_video(args):
    id_array = []
    img_array = []

    for filename in sorted( glob.glob(args.input_dir + '*.png'),
                key=lambda s: int((((os.path.basename(s).split('.')[0])).split('t'))[2])):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    
    out = cv2.VideoWriter(args.output_dir+'project.mp4',cv2.VideoWriter_fourcc(*'MP4V'), int(args.fps), size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

# fps = read_time_stamp("./raw_test/2011_09_26/2011_09_26_drive_0002_sync/image_02/timestamps.txt")
output_video(args)