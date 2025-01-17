import os, cv2
import numpy as np
import PIL
import imageio
from PIL import Image
from copy import deepcopy
from torchvision import transforms as T

XAI_ROOT = "./xai"

### ############################## ###
### EXPLANATIONS SUPPORT FUNCTIONS ###
### ############################## ###
def create_image_grid(
        crop_size:int, 
        overlap:int, 
        img:PIL.Image):
    """
    Args:
        crop_size (int): size of the (square) crop
        overlap (int): overlap (n_pixels) between adjacent crops
        img (PIL.Image): image to be cropped
    
    Returns:
        grid_dict (dict): dictionary with keys as the coordinates of the grid and values as the coordinates of the crop
        num_cols (int): number of columns in the grid
        num_rows (int): number of rows in the grid
    """
    img_w, img_h = img.size
    num_cols = (img_w - overlap)//(crop_size - overlap)
    num_rows = (img_h - overlap)//(crop_size - overlap)

    grid_w = (num_cols - 1)*(crop_size - overlap) + crop_size
    grid_h = (num_rows - 1)*(crop_size - overlap) + crop_size

    UL_x = int((img_w - grid_w)/2)
    UL_y = int((img_h - grid_h)/2)

    grid_dict = {}

    for i in range(num_rows):
        for j in range(num_cols):
            UL_x_grid = UL_x + j*(crop_size - overlap)
            UL_y_grid = UL_y + i*(crop_size - overlap)
            BR_x_grid = UL_x_grid + crop_size
            BR_y_grid = UL_y_grid + crop_size
            grid_dict[f'{i}_{j}'] = (UL_x_grid, UL_y_grid, BR_x_grid, BR_y_grid)
    
    return grid_dict, num_cols, num_rows

def generate_instance_mask(
        inst_width: int,
        inst_height: int,
        block_width: int,
        block_height: int):

    num_columns, num_rows = int(inst_width/block_width) + 1, int(inst_height/block_height) + 1
    
    idx = np.arange(int(num_columns*num_rows), dtype=np.uint16)
    np.random.shuffle(idx)
    mask = idx.reshape((num_rows, num_columns)).repeat(block_height, axis = 0).repeat(block_width, axis = 1)*1000

    mask_img = Image.fromarray(mask)
    mask_width, mask_height = mask_img.size

    left, right = (mask_width - inst_width)/2, (mask_width + inst_width)/2
    top, bottom = (mask_height - inst_height)/2, (mask_height + inst_height)/2

    mask = mask_img.crop((left, top, right, bottom))
    mask = mask.crop((0, 0, inst_width, inst_height))

    mask.save(f"{XAI_ROOT}/def_mask_{block_width}x{block_height}.png")


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
        page_img = Image.open(f"{XAI_ROOT}/data/{file_name}.jpg")
    except:
        print(f"'{file_name}' not found in './data' directory.")

    try:
        mask_img = Image.open(f"{XAI_ROOT}/explanations/crop_level/{file_name}/{file_name}_mask_blocks_{block_width}x{block_height}.png")    
    except:
        print(f"'{file_name}_mask_blocks_{block_width}x{block_height}.png' not found in './explanations/crop_level/{file_name}' directory.")

    crop_size, overlap = crop_size, overlap

    GD, NC, NR = create_image_grid(crop_size, overlap, mask_img)

    img_array = np.array(deepcopy(page_img).convert('RGB'))[:, :, ::-1]
    mask_array = np.array(deepcopy(mask_img).convert('RGB'))[:, :, ::-1]

    list_images, list_masks = list(), list()

    if not os.path.exists(f"{XAI_ROOT}/explanations/crop_level/{file_name}/crops"):
        os.mkdir(f"{XAI_ROOT}/explanations/crop_level/{file_name}/crops")
    
    for i in range(NR):
        for j in range(NC):
            x0, y0, x1, y1 = GD[f'{i}_{j}']
            
            img_crop = page_img.crop((x0, y0, x1, y1))
            img_crop.save(f"{XAI_ROOT}/explanations/crop_level/{file_name}/crops/{file_name}_{crop_size}_{overlap}_{i}_{j}.jpg")

            mask_crop = mask_img.crop((x0, y0, x1, y1))
            mask_crop.save(f"{XAI_ROOT}/explanations/crop_level/{file_name}/crops/{file_name}_mask_blocks_{block_width}x{block_height}_{crop_size}_{overlap}_{i}_{j}.png")

            img_array_copy = deepcopy(img_array)
            mask_array_copy = deepcopy(mask_array)

            cv2.rectangle(img_array_copy, (x0, y0), (x1, y1), (0, 0, 255), 5)
            cv2.rectangle(mask_array_copy, (x0, y0), (x1, y1), (0, 0, 255), 5)

            list_images.append(img_array_copy[:, :, ::-1])
            list_masks.append(mask_array_copy[:, :, ::-1])
    
    # Saves a GIF file which describes the cropping process
    imageio.mimsave(f'{XAI_ROOT}/explanations/crop_level/{file_name}/{file_name}_crops.gif', list_images, duration=1.25)

def get_instances_to_explain(dataset, source, class_to_idx, phase):
    instances, labels = list(), list()
    
    for f in os.listdir(source):
        if phase == "train":
            if (dataset == "CEDAR_Letter") and ("c" in f): continue
            if (dataset == "CVL") and ("-3" in f or "-7" in f): continue
            if (dataset == "VatLat653") and ("t" in f): continue
        if phase == "test":
            if (dataset == "CEDAR_Letter") and ("c" not in f): continue
            if (dataset == "CVL") and ("-3" not in f and "-7" not in f): continue
            if (dataset == "VatLat653") and ("t" not in f): continue
        
        writer_id = int(f[0:4])
        label = class_to_idx[str(writer_id)]
        
        src_path, dest_path = f"{source}/{f}", f"{XAI_ROOT}/data/{f}"
        os.system(f"cp {src_path} {dest_path}")
        
        instances.append(f)
        labels.append(label)
    
    return instances, labels