import random, os, cv2, imageio, torch, json
import pickle as pkl
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from captum.attr import visualization as viz
from PIL import Image
from copy import deepcopy
from torchvision.transforms import v2

from utils import get_page_dimensions

XAI_ROOT = "./xai"

### ################ ###
### IMAGE TRANSFORMS ###
### ################ ###
def create_image_grid(num_rows, num_cols, crop_size, overlap):
    grid_dict = dict()
    step = crop_size - overlap

    for i in range(0, num_rows):
        for j in range(num_cols):
            left, top = j * step, i * step
            right = left + crop_size
            bottom = top + crop_size

            grid_dict[f"{i}_{j}"] = (left, top, right, bottom)
    
    return grid_dict

def produce_padded_page(img, instance_name, instance_type, crop_size, overlap, mean, output_dir):
    img_w, img_h = img.size
    step = crop_size - overlap

    num_cols = ((img_w - overlap) // step) + 1
    num_rows = ((img_h - overlap) // step) + 1

    new_w = ((num_cols - 1) * step) + crop_size
    new_h = ((num_rows - 1) * step) + crop_size

    df_x = new_w - img_w
    df_x_left, df_x_right = 0, 0
    if df_x % 2 == 0: df_x_left = df_x_right = int(df_x/2)
    else:
        df_x_left = int((df_x-1)/2)
        df_x_right = int((df_x+1)/2)
    
    df_y = new_h - img_h
    df_y_top, df_y_bottom = 0, 0
    if df_y % 2 == 0: df_y_top = df_y_bottom = int(df_y/2)
    else:
        df_y_top = int((df_y-1)/2)
        df_y_bottom = int((df_y+1)/2)

    mean_int = [m*255 for m in mean]
    padded_img = T.Pad((df_x_left, df_y_top, df_x_right, df_y_bottom), fill=mean_int)(img)

    padded_img.save(f"{output_dir}/{instance_name}_forexp{instance_type}")
    padding_dict = {"left": df_x_left, "top": df_y_top, "right": df_x_right, "bottom": df_y_bottom}
    with open(f"{output_dir}/padding_dict.json", "w") as f: json.dump(padding_dict, f)

    return padded_img, num_rows, num_cols

def generate_mask(padded_img, dataset, patch_width, patch_height, crop_size, overlap):
    img_w, img_h = padded_img.size
    num_cols, num_rows = (img_w // patch_width) + 1, (img_h // patch_height) + 1
    total_patches = num_cols*num_rows

    idx = np.arange(total_patches, dtype=np.uint16)
    np.random.shuffle(idx)
    mask_array = idx.reshape((num_rows, num_cols)).repeat(patch_height, axis=0).repeat(patch_width, axis=1)

    mask_array_to_img = mask_array.copy()
    mask_img = Image.fromarray(mask_array_to_img*100)
    mask_img = v2.CenterCrop((img_h, img_w))(mask_img)

    output_dir = f"{XAI_ROOT}/masks/{dataset}_mask_{patch_width}x{patch_height}_cs{crop_size}_overlap{overlap}"
    mask_img.save(f"{output_dir}/mask.png")
    dimensions = {"mask_rows": num_rows, "mask_cols": num_cols}
    with open(f"{output_dir}/dimensions.json", "w") as f: json.dump(dimensions, f, indent=4)

    return mask_img, num_rows, num_cols

### ##################### ###
### POPULATION STATISTICS ###
### ##################### ###
def load_rgb_mean_std(root):
    stats = list()
    with open(root + os.sep + 'rgb_train_stats.pkl', 'rb') as f:
        stats = pkl.load(f)
    
    mean_, std_ = stats[0], stats[1]
    return mean_, std_