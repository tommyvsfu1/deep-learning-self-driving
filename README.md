# deep-learning-self-driving
<img src="raw_result/gif/1.gif" height="200">
<img src="raw_result/gif/2.gif" height="200">


## Segmentation Training
### data
use KITTI road data (I only use these for training), so this training will not change
### file form:
```
./data
    |train.csv
    |test.csv
    |data_road/
```
### how to run?
```python
python train --input_dir=<> --train_img_size=<> output_dir=<>
```
>- input_dir='./data/' (for loader to load csv, and training data)
>- train_img_size=160,576
>- output_dir=<fill in>
### Where does model save
```
./checkpoint
```
## Segmentation Evaluation
### data
KITTI raw data or testing
### how to run?
```python
python eval.py --input_dir=<> --test_img_size=<> output_dir=<>
```
Sample:
```
python eval.py --input_dir='./raw_test/purpose/image_02/' --test_img_size=256,1024 --output_dir='./this_is_test'
```
>- input_dir='./data/' (for loader to load csv, and training data)
>- train_img_size=256,1024
>- output_dir=<fill in>

### model load from
```
./checkpoint/pretrained/best_seg.cpt
```

## Detection with Segemtation Evaluation
### data
KITTI raw data or testing

### run
```
cd reference_code/PyTorch-YOLOv3/
python detect_seg.py --mask_folder='../../raw_test/purpose/result2/' --image_folder='../../raw_test/purpose/result1/' --output_folder='../../segdect/'
```
## Video

### run
```python
python video_capture.py --input_dir=<> --fps=<> --output_dir=<>
```
sample:
```python
python video_capture.py --input_dir='./raw_test/purpose/result3/' --fps=15 --output_dir='./'
```


### ALL IN ONE


TODO:  
1.train/ testing / raw_testing option  
2.csv generator  (now in utils)
3.video generator   


for now:  
1. download raw video(image)  
2. generate csv file(for dataset loader)  
3. use raw_testing function to generate image with segmask  
4. use video_capture.py to calculate fps  
5. use video_capture.py to generate video  

(use generate.sh)



## ignore
data/data_road/ -> KITTI road data
data/training_mapping/ -> no use
data/devkit_road/ -> no use
data/data_road.zip -> no use
reference_code/ 
result/ -> no use
__pycache__/ -> no use
nohup.out -> no use
scores/ -> no use
raw_test/ -> KITTI raw data
raw_result/raw/ -> KITTI raw data result