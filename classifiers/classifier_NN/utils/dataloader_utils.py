import torch, glob, os, cv2, numbers
import numpy as np
import pickle as pkl
import torchvision.datasets as datasets
import torchvision.transforms as T
import torchvision.transforms.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

LOG_ROOT = "./log"

### ############## ###
### DATASETS STATS ###
### ############## ###
def compute_mean_and_std(root):

    types = ('*.png', '*.jpg')
    training_images = []
    for files in types:
        training_images.extend(glob.glob(root + '/*/' + files))	

    pixel_num = 0
    channel_sum, channel_sum_squared = np.zeros(3), np.zeros(3)
    
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
    with open(root + os.sep + 'rgb_train_stats.pkl', 'wb') as f:
        pkl.dump(stats, f) 

    return rgb_mean, rgb_std

def load_rgb_mean_std(root):
    try:
        stats = list()
        with open(root + os.sep + 'rgb_train_stats.pkl', 'rb') as f:
            stats = pkl.load(f)
        
        mean_, std_ = stats[0], stats[1]
    except: mean_, std_ = compute_mean_and_std(root = root)

    return mean_, std_

### ################# ###
### CUSTOM TRANSFORMS ###
### ################# ###

### Invert ###
class Invert(object):
    def __call__(self, x):
        x = F.invert(x)
        return x
    
    def __str__(self):
        str_transforms = f"Invert RGB channels"
        return str_transforms

### AddGaussianNoise ###
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        randn_tensor = torch.randn(1, tensor.size()[1], tensor.size()[2])
        randn_tensor = randn_tensor.repeat(3,1,1)
        return tensor + randn_tensor * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

### NRandomCrop ###
class NRandomCrop(object):
    def __init__(self, crop_size, number=1, padding=0, pad_if_needed=False):
        if isinstance(crop_size, numbers.Number): self.crop_size = (int(crop_size), int(crop_size))
        else: self.crop_size = crop_size
        
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.number = number

    @staticmethod
    def get_params(img, output_size, number):
        img_w, img_h = img.size
        crop_w, crop_h = output_size
        
        if img_w == crop_w and img_h == crop_h:
            return 0, 0, crop_w, crop_h
        
        x_list = np.random.randint(low=0, high=img_w - crop_w + 1, size=number).tolist()
        y_list = np.random.randint(low=0, high=img_h - crop_h + 1, size=number).tolist()

        return x_list, y_list, crop_w, crop_h

    def __call__(self, img):
        img_w, img_h = img.size
        crop_w, crop_h = self.crop_size
        
        if self.padding > 0:
            img = T.Pad(self.padding, padding_mode="edge")(img)
        
        if self.pad_if_needed and img_w < crop_w:
            padding_x = int((crop_w - img_w) // 2)
            img = T.Pad((padding_x, 0, padding_x, 0), padding_mode="edge")(img)
            img_w = img.size[0]
        
        if self.pad_if_needed and img_h < crop_h:
            padding_y = int((crop_h - img_h) // 2)
            img = T.Pad((0, padding_y, 0, padding_y), padding_mode="edge")(img)
            img_h = img.size[1]
        
        x_list, y_list, crop_w, crop_h = self.get_params(img, self.crop_size, self.number)

        return n_random_crops(img, x_list, y_list, crop_w, crop_h)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.crop_size, self.padding)

def n_random_crops(img, x_list, y_list, crop_w, crop_h):
    crops = list()
    
    for i in range(0, len(x_list)):
        left, top = x_list[i], y_list[i]
        right, bottom = left + crop_w, top + crop_h
        crops.append(img.crop((left, top, right, bottom)))
    
    return tuple(crops)

### AllCrops ###
class AllCrops(object):
    def __init__(self, size, mult_factor):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.mult_factor = mult_factor
        
    def __call__(self, img):
        return all_crops(img, self.size, self.mult_factor)

def all_crops(img, size, mult_factor):
    crops = list()
    img_w, img_h = img.size
    crop_w, crop_h = size
    
    num_crops_x = mult_factor * (1 + (img_w // crop_w))
    num_crops_y = mult_factor * (1 + (img_h // crop_h))
    
    overlap_x = (num_crops_x * crop_w - img_w) // max(1, num_crops_x - 1)
    overlap_y = (num_crops_y * crop_h - img_h) // max(1, num_crops_y - 1)
    
    for i in range(0, num_crops_x):
        for j in range(0, num_crops_y):
            left = i * (crop_w - overlap_x)
            top = j * (crop_h - overlap_y)
            right = left + crop_w
            bottom = top + crop_h

            crops.append(img.crop((left, top, right, bottom)))

    return crops


### ########### ###
### DATALOADERS ###
### ########### ###

### Superclass ###
class Base_DataLoader():
    def __init__(self, directory, classes, batch_size, img_crop_size, mult_factor=1, weighted_sampling=True, phase='train', mean=[0, 0, 0], std=[1, 1, 1], shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.classes = classes
        self.img_crop_size = img_crop_size
        self.mult_factor = mult_factor
        self.weighted_sampling = weighted_sampling
        self.phase = phase
        self.shuffle = shuffle
        self.mean = mean
        self.std = std
    
    def make_weights_for_balanced_classes(self, images, nclasses):
        count = [0] * nclasses
        for item in images:
            count[item[1]] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]
        return weight

    def generate_dataset(self):
        ds = datasets.ImageFolder(root=self.directory, transform=self.compose_transform())
        
        class_to_idx, c_id = dict(), 0
        for c in self.classes:
            class_to_idx[c] = c_id
            c_id += 1
        
        ds.class_to_idx = class_to_idx
        
        return ds
    
    def load_data(self):
        num_workers = int(os.cpu_count()/2)
        dataset = self.generate_dataset()
        if self.phase == 'train' and self.weighted_sampling:
            weights = self.make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes))
            weights = torch.DoubleTensor(weights)
            sampler = WeightedRandomSampler(weights, len(weights))
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=num_workers)
        else:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=num_workers)
        return dataset, loader

### Subclasses ###
class Train_Test_DataLoader(Base_DataLoader):
    def compose_transform(self, 
    cjitter = {'brightness': [0.4, 1.3], 'contrast': 0.6, 'saturation': 0.6,'hue': 0.4}, 
    cjitter_p = 1, 
    randaffine = {'degrees': [-10,10], 'translate': [0.2, 0.2], 'scale': [1.3, 1.4], 'shear': 1}, 
    randpersp = {'distortion_scale': 0.1, 'p': 0.2}, 
    gray_p = 0.2, 
    gaussian_blur = {'kernel_size': 3, 'sigma': [0.1, 0.5]},
    rand_eras = {'p': 0.33, 'scale': [0.02, 0.33], 'ratio': [0.3, 3.3]},
    invert_p = 0.05,
    gaussian_noise = {'mean': 0., 'std': 0.004},
    gn_p = 0.0):

        randaffine['interpolation'] = randpersp['interpolation'] = T.InterpolationMode.BILINEAR
        randaffine['fill'] = randpersp['fill'] = [255, 255, 255]
        rand_eras['value'] = self.mean
                
        train_transforms = T.Compose([
            AllCrops(size=self.img_crop_size, mult_factor=self.mult_factor),
            T.Lambda(lambda crops: torch.stack([
                T.RandomApply([AddGaussianNoise(**gaussian_noise)], p=gn_p)
                (T.Normalize(self.mean, self.std)
                (T.RandomApply([Invert()], p=invert_p)
                (T.RandomErasing(**rand_eras)
                (T.ToTensor()
                (T.RandomGrayscale(gray_p)
                (T.GaussianBlur(**gaussian_blur)
                (T.RandomPerspective(**randpersp)
                (T.RandomAffine(**randaffine)
                (T.RandomApply([T.ColorJitter(**cjitter)], p=cjitter_p)(crop)))))))))) for crop in crops]))
        ])
        
        val_transforms = T.Compose([
            NRandomCrop(crop_size = self.img_crop_size, number = 15, pad_if_needed = True),
            T.Lambda(lambda crops: torch.stack([T.Normalize(self.mean, self.std)(T.ToTensor()(crop)) for crop in crops]))
        ])
        
        test_transforms = T.Compose([
            NRandomCrop(crop_size = self.img_crop_size, number = 150, pad_if_needed = True),
            T.Lambda(lambda crops: torch.stack([T.Normalize(self.mean, self.std)(T.ToTensor()(crop)) for crop in crops]))
        ])
        
        if self.phase == "train": return train_transforms
        if self.phase == "val": return val_transforms
        if self.phase == "test": return test_transforms

class Confidence_Test_DataLoader(Base_DataLoader):
    def compose_transform(self, n_pair_test_crops = 250):

        pair_test_scan_transforms = T.Compose([
            AllCrops(size = self.img_crop_size, mult_factor=self.mult_factor),
            T.Lambda(lambda crops: torch.stack([T.Normalize(self.mean, self.std)(T.ToTensor()(crop)) for crop in crops]))
        ])
    
        pair_test_random_transforms = T.Compose([
            NRandomCrop(crop_size = self.img_crop_size, number = n_pair_test_crops, pad_if_needed = True),
            T.Lambda(lambda crops: torch.stack([T.Normalize(self.mean, self.std)(T.ToTensor()(crop)) for crop in crops]))
        ])
            
        if self.phase == "scan": return pair_test_scan_transforms
        if self.phase == "random": return pair_test_random_transforms   