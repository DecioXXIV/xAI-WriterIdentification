import os
import numpy as np
from PIL import Image

def generate_page_mask(
        file_name:str, 
        block_height:int, 
        block_width:int, 
        verbose:bool=True):
    """
    Args:
        file_name (str): name of the image file (without extension) to be cropped
        block_height (int), block_width (int): dimensions of the mask blocks
        verbose (bool): print image sizes if True

    Returns:
        None -> saves the mask in the './data/<file_name>' directory
    """

    img = Image.open(f"./data/{file_name}.jpg")
    img_width, img_height = img.size

    if verbose:
        print(f"Original Image Size: {img_width}x{img_height} pixels")

    block_width, block_height = block_width, block_height
    num_columns, num_rows = int(img_width/block_width) + 1, int(img_height/block_height) + 1

    idx = np.arange(int(num_columns*num_rows), dtype=np.uint16)
    np.random.shuffle(idx)
    mask = idx.reshape((num_rows, num_columns)).repeat(block_height, axis = 0).repeat(block_width, axis = 1)*1000

    mask_img = Image.fromarray(mask)
    mask_width, mask_height = mask_img.size

    left, right = (mask_width - img_width)/2, (mask_width + img_width)/2
    top, bottom = (mask_height - img_height)/2, (mask_height + img_height)/2

    mask = mask_img.crop((left, top, right, bottom))
    mask = mask.crop((0, 0, img_width, img_height))
    if verbose:
        mask_width, mask_height = mask.size
        print(f"Mask Image Size: {mask_width}x{mask_height} pixels")
    
    if not os.path.exists(f"./data/{file_name}"):
        os.mkdir(f"./data/{file_name}")

    mask.save(f"./data/{file_name}/{file_name}_mask_blocks_{block_width}x{block_height}.png")