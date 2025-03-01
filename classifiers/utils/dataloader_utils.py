import torch, glob, os, cv2, numbers
import numpy as np
import pickle as pkl
import torchvision.datasets as datasets
import torchvision.transforms as T
import torchvision.transforms.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

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
    def __init__(self, crop_size, number, random_seed=24):
        if isinstance(crop_size, numbers.Number): self.crop_size = (int(crop_size), int(crop_size))
        else: self.crop_size = crop_size
        self.number = number
        self.random_seed = random_seed

    def __call__(self, img):
        return n_random_crops(img, self.number, self.crop_size, self.random_seed)

def n_random_crops(img, number, crop_size, random_seed=24):
    crops = list()
    img_w, img_h = img.size
    crop_w, crop_h = crop_size
    
    np.random.seed(random_seed)
    lefts = np.random.randint(low=0, high=img_w - crop_w + 1, size=number).tolist()
    tops = np.random.randint(low=0, high=img_h - crop_h + 1, size=number).tolist()
    
    for i in range(0, number):
        crop = img.crop((lefts[i], tops[i], lefts[i] + crop_w + 1, tops[i] + crop_h + 1))
        crops.append(crop)
    
    return crops
    
### AllCrops ###
class AllCrops(object):
    def __init__(self, crop_size, mult_factor):
        if isinstance(crop_size, numbers.Number): self.crop_size = (int(crop_size), int(crop_size))
        else: self.crop_size = crop_size
        self.mult_factor = mult_factor
        
    def __call__(self, img):
        return all_crops(img, self.crop_size, self.mult_factor)

def all_crops(img, crop_size, mult_factor):
    crops = list()
    img_w, img_h = img.size
    crop_w, crop_h = crop_size
    
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
            
            right_pad, bottom_pad = 0, 0
            if right > img_w:
                right_pad = right - img_w
                right = img_w
            if bottom > img_h:
                bottom_pad = bottom - img_h
                bottom = img_h
            
            crop = img.crop((left, top, right, bottom))
            crop = T.Pad((0, 0, right_pad, bottom_pad), padding_mode="edge")(crop)

            crops.append(crop)

    return crops

### ########### ###
### DATALOADERS ###
### ########### ###
class Base_DataLoader():
    def __init__(self, directory, classes, batch_size, img_crop_size, mean=[0, 0, 0], std=[1, 1, 1]):
        self.directory = directory
        self.classes = classes
        self.batch_size = batch_size
        self.img_crop_size = img_crop_size
        self.mean = mean
        self.std = std
    
    def generate_dataset(self):
        ds = datasets.ImageFolder(root=self.directory, transform=self.compose_transform())
        
        class_to_idx, c_id = dict(), 0
        for c in self.classes:
            class_to_idx[c] = c_id
            c_id += 1
        
        ds.class_to_idx = class_to_idx
        
        return ds

class Train_DataLoader(Base_DataLoader):
    def __init__(self, directory, classes, batch_size, img_crop_size, mult_factor=1, weighted_sampling=True, mean=[0, 0, 0], std=[1, 1, 1], shuffle=True):
        super().__init__(directory, classes, batch_size, img_crop_size, mean, std)
        self.mult_factor = mult_factor
        self.weighted_sampling = weighted_sampling
        self.shuffle = shuffle
    
    def make_weights_for_balanced_classes(self, images, nclasses):
        count = [0] * nclasses
        for item in images: count[item[1]] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses): weight_per_class[i] = N / float(count[i])
        weight = [0] * len(images)
        for idx, val in enumerate(images): weight[idx] = weight_per_class[val[1]]
        return weight
    
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
        
        transforms = T.Compose([
            T.RandomApply([T.ColorJitter(**cjitter)], p=cjitter_p),
            T.RandomAffine(**randaffine),
            T.RandomPerspective(**randpersp),
            T.GaussianBlur(**gaussian_blur),
            T.RandomGrayscale(gray_p),
            T.ToTensor(),
            T.RandomErasing(**rand_eras),
            T.RandomApply([Invert()], p=invert_p),
            T.Normalize(self.mean, self.std),
            T.RandomApply([AddGaussianNoise(**gaussian_noise)], p=gn_p)
        ])
        
        return transforms
    
    def load_data(self):
        num_workers = max(1, int(os.cpu_count()/2))
        dataset = self.generate_dataset()
        weights = self.make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes))
        weights = torch.DoubleTensor(weights)
        sampler = WeightedRandomSampler(weights, len(weights))
        loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=num_workers)
        
        return dataset, loader

class Test_DataLoader(Base_DataLoader):
    def __init__(self, directory, classes, batch_size, img_crop_size, mean=[0, 0, 0], std=[1, 1, 1]):
        super().__init__(directory, classes, batch_size, img_crop_size, mean, std)
    
    def compose_transform(self):
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])
        
        return transforms
    
    def load_data(self):
        num_workers = max(1, int(os.cpu_count()/2))
        dataset = self.generate_dataset()
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers)
        
        return dataset, loader

class Confidence_Test_DataLoader(Test_DataLoader):
    def __init__(self, directory, classes, batch_size, img_crop_size, mode, mult_factor=1, n_pair_test_crops=250, random_seed=24, mean=[0, 0, 0], std=[1, 1, 1]):
        super().__init__(directory, classes, batch_size, img_crop_size, mean, std)
        self.mode = mode
        self.mult_factor = mult_factor
        self.n_pair_test_crops = n_pair_test_crops
        self.random_seed = random_seed
    
    def compose_transform(self):
        pair_test_scan_transforms = T.Compose([
            AllCrops(size = self.img_crop_size, mult_factor=self.mult_factor),
            T.Lambda(lambda crops: torch.stack([T.Normalize(self.mean, self.std)(T.ToTensor()(crop)) for crop in crops]))
        ])
    
        pair_test_random_transforms = T.Compose([
            NRandomCrop(crop_size = self.img_crop_size, number = self.n_pair_test_crops, random_seed=self.random_seed),
            T.Lambda(lambda crops: torch.stack([T.Normalize(self.mean, self.std)(T.ToTensor()(crop)) for crop in crops]))
        ])
            
        if self.mode == "scan": return pair_test_scan_transforms
        if self.mode == "random": return pair_test_random_transforms
        
class Faithfulness_Test_DataLoader(Test_DataLoader):
    def __init__(self, directory, classes, batch_size, img_crop_size, n_pair_test_crops=250, random_seed=24, mean=[0, 0, 0], std=[1, 1, 1]):
        super().__init__(directory, classes, batch_size, img_crop_size, mean, std)
        self.n_pair_test_crops = n_pair_test_crops
        self.random_seed = random_seed
    
    def compose_transform(self):    
        transforms = T.Compose([
            NRandomCrop(crop_size = self.img_crop_size, number = self.n_pair_test_crops, random_seed=self.random_seed),
            T.Lambda(lambda crops: torch.stack([T.Normalize(self.mean, self.std)(T.ToTensor()(crop)) for crop in crops]))
        ])
            
        return transforms