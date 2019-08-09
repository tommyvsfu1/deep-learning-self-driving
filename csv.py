
import argparse
import os

def get_csv_test_file(data_folder):
    image_paths = sorted(glob(os.path.join(data_folder, '*.png')))

    import csv

    # 開啟輸出的 CSV 檔案
    with open(opt.output_csv_folder+'raw_test.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        for image in image_paths:
            writer.writerow([os.path.basename(image)])

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, required=True,help="path to checkpoint model")
parser.add_argument("--output_csv_folder", type=str, required=True,help="path to checkpoint model")

opt = parser.parse_args()
get_csv_test_file(opt.input_folder)