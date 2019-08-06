#/bin/bash
python resize.py --image_folder='./raw_test/purpose/image_02/data/' --out_folder='./raw_test/purpose/result1/'

python train.py --output_dir='./raw_raw_result'

python detect_seg.py --image_folder='../../raw_test/purpose/result1/' --mask_folder='../../raw_test/purpose/result2/'

python video_capture.py
