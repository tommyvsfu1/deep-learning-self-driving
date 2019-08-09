#/bin/bash
raw_dir="raw_test/purpose/image_02/"
raw_img=$raw_dir"data/"
resize_dir="./resize_out/"
resize_img=$resize_dir"data/"


# segmentation
seg_img=$resize_dir"seg/"

# detection
detect_seg="../../$seg_img"
detect_img="../../"$resize_img
out_dir="../../"$resize_dir"result/"

video_in=$resize_dir"result/"
fps="15"
video_out=$resize_dir

#resize image
python resize.py --input_folder=$raw_img --img_size=256,1024 --output_folder=$resize_img
# generate csv
python csv_builder.py --input_folder=$resize_img --output_csv_folder=$resize_dir

echo "segmentation start"
#segmentation
cd ~/deep-learning-self-driving
echo $resize_dir
echo $seg_img
python eval.py --input_dir=$resize_dir --test_img_size=256,1024 --output_dir=$seg_img
detection using resize image and segmentation result
cd reference_code/PyTorch-YOLOv3/
python detect_seg.py --mask_folder=$detect_seg --image_folder=$detect_img --output_folder=$out_dir
#make video 
cd ~/deep-learning-self-driving
python video_capture.py --input_dir=$video_in --fps=$fps --output_dir=$video_out