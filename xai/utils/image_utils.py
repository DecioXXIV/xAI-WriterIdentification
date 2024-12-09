import random, os, glob, cv2
import pickle as pkl
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from captum.attr import visualization as viz

### ################ ###
### IMAGE TRANSFORMS ###
### ################ ###
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