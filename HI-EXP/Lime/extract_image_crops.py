import os
import numpy as np
import cv2
import imageio
from PIL import Image
from copy import deepcopy
from create_image_grid import create_image_grid

def extract_image_crops(
        file_name:str, 
        block_width:int, 
        block_height:int, 
        crop_size:int, 
        overlap:int):
    """
    Args:
        file_name (str): name of the image file (without extension) to be cropped
        block_width (int), block_height (int): dimensions of the mask blocks
        crop_size (int): size of the (square) crop
        overlap (int): overlap (n_pixels) between adjacent crops
    
    Returns:
        None -> saves the cropped images and masks in the './data/<file_name>/crops' directory
    """
    
    try:
        page_img = Image.open(f"./data/{file_name}.jpg")
    except:
        print(f"'{file_name}' not found in './data' directory.")

    try:    
        mask_img = Image.open(f"./data/{file_name}/{file_name}_mask_blocks_{block_width}x{block_height}.png")
    except:
        print(f"'{file_name}_mask_blocks_{block_width}x{block_height}.png' not found in './data/{file_name}' directory.")

    crop_size, overlap = crop_size, overlap

    GD, NC, NR = create_image_grid(crop_size, overlap, mask_img)

    img_array = np.array(deepcopy(page_img).convert('RGB'))[:, :, ::-1]
    mask_array = np.array(deepcopy(mask_img).convert('RGB'))[:, :, ::-1]

    list_images, list_masks = list(), list()

    if not os.path.exists(f"./data/{file_name}/crops"):
        os.mkdir(f"./data/{file_name}/crops")

    for i in range(NR):
        for j in range(NC):
            x0, y0, x1, y1 = GD[f'{i}_{j}']
            
            img_crop = page_img.crop((x0, y0, x1, y1))
            img_crop.save(f"./data/{file_name}/crops/{file_name}_{crop_size}_{overlap}_{i}_{j}.jpg")

            mask_crop = mask_img.crop((x0, y0, x1, y1))
            mask_crop.save(f"./data/{file_name}/crops/{file_name}_mask_blocks_{block_width}x{block_height}_{crop_size}_{overlap}_{i}_{j}.png")

            img_array_copy = deepcopy(img_array)
            mask_array_copy = deepcopy(mask_array)

            cv2.rectangle(img_array_copy, (x0, y0), (x1, y1), (0, 0, 255), 5)
            cv2.rectangle(mask_array_copy, (x0, y0), (x1, y1), (0, 0, 255), 5)

            list_images.append(img_array_copy[:, :, ::-1])
            list_masks.append(mask_array_copy[:, :, ::-1])
    
    # Saves a GIF file which describes the cropping process
    imageio.mimsave(f'./data/{file_name}/crops/{file_name}_crops.gif', list_images, duration=0.8)