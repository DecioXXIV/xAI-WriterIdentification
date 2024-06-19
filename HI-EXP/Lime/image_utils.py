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
    
    mask_to_tensor = T.ToTensor()

    img = img_transforms(img)
    
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
    try:
        stats = []
        
        with open(root + os.sep + 'rgb_stats.pkl', 'rb') as f:
            stats = pkl.load(f)
        
        mean_ = stats[0]
        std_ = stats[1]
    except:
        mean_, std_ = compute_mean_and_std(root = root)

    return mean_, std_

def compute_mean_and_std(root):

	types = ('*.png', '*.jpg')
	training_images = []
	for files in types:
		training_images.extend(glob.glob(root + '/*/' + files))	

	pixel_num = 0
	channel_sum = np.zeros(3)
	channel_sum_squared = np.zeros(3)

	for i in tqdm(training_images):
		im = cv2.imread(i)
		im = im/255.0

		pixel_num += (im.size/3)
		channel_sum += np.sum(im, axis = (0, 1))
		channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

	bgr_mean = channel_sum/pixel_num
	bgr_std = np.sqrt(channel_sum_squared/pixel_num - np.square(bgr_mean))

	# change the format from bgr to rgb
	rgb_mean = list(bgr_mean)[::-1]
	rgb_std = list(bgr_std)[::-1]

	stats = [rgb_mean, rgb_std]
	with open(root + os.sep + 'rgb_stats.pkl', 'wb') as f:
		pkl.dump(stats, f) 

	return rgb_mean, rgb_std

### ############# ###
### VISUALIZATION ###
### ############# ###
def show_attr(attr_map, output_name):
    fig, _ = viz.visualize_image_attr(
        # attr_map, #.permute(1, 2, 0).cpu().numpy(),  # adjust shape to height, width, channels
        attr_map.permute(1, 2, 0).cpu().numpy(), 
        method='heat_map',
        sign='all',
        show_colorbar=True,
        use_pyplot = False
    )

    fig.savefig(f'{output_name}_att_heat_map.png')