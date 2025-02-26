import random, os, cv2, imageio, torch
import pickle as pkl
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from captum.attr import visualization as viz
from PIL import Image
from copy import deepcopy

XAI_ROOT = "./xai"

### ################ ###
### IMAGE TRANSFORMS ###
### ################ ###
def create_image_grid(
        crop_size:int, 
        overlap:int, 
        img: Image):
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
        patch_width: int,
        patch_height: int):

    num_columns, num_rows = int(inst_width/patch_width) + 1, int(inst_height/patch_height) + 1
    
    idx = np.arange(int(num_columns*num_rows), dtype=np.uint16)
    np.random.shuffle(idx)
    mask = idx.reshape((num_rows, num_columns)).repeat(patch_height, axis = 0).repeat(patch_width, axis = 1)*1000

    mask_img = Image.fromarray(mask)
    mask_width, mask_height = mask_img.size

    left, right = (mask_width - inst_width)/2, (mask_width + inst_width)/2
    top, bottom = (mask_height - inst_height)/2, (mask_height + inst_height)/2

    mask = mask_img.crop((left, top, right, bottom))
    mask = mask.crop((0, 0, inst_width, inst_height))

    mask.save(f"{XAI_ROOT}/def_mask_{patch_width}x{patch_height}.png")

def get_crops_bbxs(image, crop_width, crop_height):
    crop_bbxs = list()
    img_width, img_height = image.size
    
    num_crops_x = 1 + (img_width // crop_width)
    num_crops_y = 1 + (img_height // crop_height)
    
    overlap_x = (num_crops_x * crop_width - img_width) // max(1, num_crops_x - 1)
    overlap_y = (num_crops_y * crop_height - img_height) // max(1, num_crops_y - 1)
    
    for i in range(0, num_crops_x):
        for j in range(0, num_crops_y):
            left = i * (crop_width - overlap_x)
            top = j * (crop_height - overlap_y)
            right = left + crop_width
            bottom = top + crop_height
            
            crop_bbxs.append((left, top, right, bottom))
    
    return crop_bbxs

def extract_image_crops(
        file_name:str, 
        patch_width:int, 
        patch_height:int, 
        crop_size:int, 
        overlap:int):
    """
    Args:
        file_name (str): name of the image file (without extension) to be cropped
        patch_width (int), patch_height (int): dimensions of the mask patchs
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
        mask_img = Image.open(f"{XAI_ROOT}/explanations/crop_level/{file_name}/{file_name}_mask_patchs_{patch_width}x{patch_height}.png")    
    except:
        print(f"'{file_name}_mask_patchs_{patch_width}x{patch_height}.png' not found in './explanations/crop_level/{file_name}' directory.")

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
            mask_crop.save(f"{XAI_ROOT}/explanations/crop_level/{file_name}/crops/{file_name}_mask_patchs_{patch_width}x{patch_height}_{crop_size}_{overlap}_{i}_{j}.png")

            img_array_copy = deepcopy(img_array)
            mask_array_copy = deepcopy(mask_array)

            cv2.rectangle(img_array_copy, (x0, y0), (x1, y1), (0, 0, 255), 5)
            cv2.rectangle(mask_array_copy, (x0, y0), (x1, y1), (0, 0, 255), 5)

            list_images.append(img_array_copy[:, :, ::-1])
            list_masks.append(mask_array_copy[:, :, ::-1])
    
    # Saves a GIF file which describes the cropping process
    imageio.mimsave(f'{XAI_ROOT}/explanations/crop_level/{file_name}/{file_name}_crops.gif', list_images, duration=1.25)

def apply_transforms_crops(img, mask, mean, std):
    img_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
        ])
    
    # img_transforms: "img" is a PIL.Image, with values in [0, 255]
        # 1. ToTensor() converts "img" from PIL.Image to torch.Tensor, with values in [0, 1]
        # 2. Normalize(mean, std) normalizes "img" channel-wise
    
    img = img_transforms(img)
    mask_to_tensor = T.ToTensor()

    # Step 1
    mask = mask_to_tensor(mask)*255
    seg_ids = sorted(mask.unique().tolist())
    
    # Step 2
    feature_mask = mask.clone()
    for i, seg_id in enumerate(seg_ids):
        feature_mask[feature_mask == seg_id] = i

    feature_mask = feature_mask.to(torch.int64)[0,:,:].unsqueeze(0)

    seg_ids = sorted(feature_mask.unique().tolist())

    return img, feature_mask, len(seg_ids)    

def apply_transforms(img, mask, img_crop_size, mean, std, output_name):
    width, height = img.size
    UL_x = random.randint(0, width - img_crop_size - 1)
    UL_y = random.randint(0, height - img_crop_size - 1)

    img.crop((UL_x, UL_y, UL_x + img_crop_size, UL_y + img_crop_size)).save(f'{output_name}_crop.png')

    img_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
        ])
    
    mask_to_tensor = T.ToTensor()

    img = img_transforms(img)
    img = F.crop(img, UL_y, UL_x, img_crop_size, img_crop_size)
    
    # Step 1
    mask_tensor_1 = mask_to_tensor(mask)
    mask = F.crop(mask_tensor_1, UL_y, UL_x, img_crop_size, img_crop_size)*255

    seg_ids = sorted(mask.unique().tolist())
    # Step 2
    feature_mask = mask.clone()
    for i, seg_id in enumerate(seg_ids):
        feature_mask[feature_mask == seg_id] = i

    feature_mask = feature_mask.to(torch.int64)[0,:,:].unsqueeze(0)

    seg_ids = sorted(feature_mask.unique().tolist())

    return img, feature_mask, len(seg_ids)

### ##################### ###
### POPULATION STATISTICS ###
### ##################### ###
def load_rgb_mean_std(root):
    stats = list()
    with open(root + os.sep + 'rgb_train_stats.pkl', 'rb') as f:
        stats = pkl.load(f)
    
    mean_, std_ = stats[0], stats[1]
    return mean_, std_

### ############# ###
### VISUALIZATION ###
### ############# ###
def show_attr(attr_map, output_name):
    fig, _ = viz.visualize_image_attr(
        attr_map.permute(1, 2, 0).cpu().numpy(), # adjust shape to height, width, channels
        method='heat_map',
        sign='all',
        show_colorbar=True,
        use_pyplot = False
    )

    fig.savefig(f'{output_name}_att_heat_map.png')