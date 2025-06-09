from tools.imageTools import Enhancement
import cv2
from glob import glob
import os
from tqdm import tqdm

image_folder = "/home/hong_data1/Documents/Delaney/yeast_viability/flowrates/run4/"
save_folder = f"{image_folder}/enh"

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

stackSize = 40
imgsz = (1440, 1080)

eh = Enhancement(imgsz=imgsz, stackSize=stackSize)
image_files = sorted(glob(f"{image_folder}/*.jpg"))

for i in tqdm(range(stackSize), desc='Warming up'):
    image = cv2.imread(image_files[i], 0)
    image = cv2.resize(image, imgsz)
    image = eh.movingWindowEnhance(image, toCPU=True)

for image_file in tqdm(image_files, desc='Enhancing'):
    image = cv2.imread(image_file, 0)
    image = cv2.resize(image, imgsz)
    image = eh.movingWindowEnhance(image, toCPU=True)
    file_name = os.path.basename(image_file)

    cv2.imwrite(f"{save_folder}/{file_name}", image)