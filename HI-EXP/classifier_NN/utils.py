import torch, glob, os, cv2, numbers, random, sys, itertools
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import pickle as pkl
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as T
import torchvision.transforms.functional as F

# Model

class BaseModel(nn.Module):
    def __init__(self,):
        super().__init__()
        encoder = models.__dict__['resnet18'](num_classes = 512)
        encoder = nn.Sequential(*(list(encoder.children())[:-1]))
        self.enc = encoder
        self.fc_layers = nn.Sequential()
        new_layer = add_fc_layer('last', 512,  1024)
        self.fc_layers.add_module('fc0', new_layer)

    def forward(self, x):
        x = self.enc(x).squeeze()
        x = self.fc_layers(x)
        return x

def load_model(mode, cp_path):
    cp = torch.load(cp_path)['model_state_dict']
    cp.pop('alpha')

    base_model = BaseModel()

    base_model.load_state_dict(cp)
   
    if mode == 'frozen':
        for param in base_model.parameters():
            param.requires_grad = False
    
    return base_model

def add_fc_layer(type_fc_layer, in_f, out_f):
    if type_fc_layer == 'last':
        fc_block = nn.Sequential(*[nn.Linear(in_features = in_f, out_features = out_f)])
    elif type_fc_layer == 'hidden':
        fc_block = nn.Sequential(*[nn.Linear(in_features = in_f, out_features = out_f),
                    nn.BatchNorm1d(out_f), nn.ReLU(), nn.Dropout(p = 0.3)])
    else:
        raise Exception('Wrong fully connected layer type')
    
    return fc_block 

class Classification_Model(nn.Module):
    def __init__(self, mode = None, cp_path = './', num_classes = 8):
        super().__init__()
        self.base_model = load_model(mode, cp_path)
        self.fc_layers = nn.Sequential()
        self.num_classes = num_classes
        new_layer = add_fc_layer('hidden', 1024,  32)
        class_layer = add_fc_layer('last', 32, self.num_classes)
        self.fc_layers.add_module('fc0', new_layer)
        self.fc_layers.add_module('fc1', class_layer)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc_layers(x)
        return x

# Dataset

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

def n_random_crops(img, x, y, h, w):

	crops = []
	for i in range(len(x)):
		new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
		crops.append(new_crop)
	return tuple(crops)

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

class Invert(object):
    def __call__(self, x):
        x = F.invert(x)
        return x
    
    def __str__(self):
        str_transforms = f"Invert RGB channels"
        return str_transforms

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

class NRandomCrop(object):

	def __init__(self, size, n=1, padding=0, pad_if_needed=False):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.padding = padding
		self.pad_if_needed = pad_if_needed
		self.n = n

	@staticmethod
	def get_params(img, output_size, n):
		w, h = img.size
		th, tw = output_size
		if w == tw and h == th:
			return 0, 0, h, w

		i_list = [random.randint(0, h - th) for i in range(n)]
		j_list = [random.randint(0, w - tw) for i in range(n)]
		return i_list, j_list, th, tw

	def __call__(self, img):
		
		if self.padding > 0:
			img = F.pad(img, self.padding)

		if self.pad_if_needed and img.size[0] < self.size[1]:
			img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))

		if self.pad_if_needed and img.size[1] < self.size[0]:
			img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

		i, j, h, w = self.get_params(img, self.size, self.n)

		return n_random_crops(img, i, j, h, w)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class Standard_DataLoader():
	
	def __init__(self, directory, batch_size, weighted_sampling = True, phase = 'train', mean = [0, 0, 0], std = [1, 1, 1], shuffle = True):

		self.directory = directory
		self.batch_size = batch_size
		self.weighted_sampling = weighted_sampling
		self.phase = phase
		self.shuffle = shuffle
		if (mean == [0, 0, 0] and std == [1, 1, 1]):
			self.mean, self.std = compute_mean_and_std(self.directory)
		else:
			self.mean = mean
			self.std = std

	def make_weights_for_balanced_classes(self, images, nclasses): # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
		count = [0] * nclasses                                                      
		for item in images:
			count[item[1]] += 1
		weight_per_class = [0.] * nclasses
		N = float(sum(count))
		for i in range(nclasses):
			weight_per_class[i] = N/float(count[i])
		weight = [0] * len(images)
		for idx, val in enumerate(images):
			weight[idx] = weight_per_class[val[1]]
		return weight    

	def compose_transform(self, 
		img_crop_size = 380,
		cjitter = {'brightness': [0.4, 1.3], 'contrast': 0.6, 'saturation': 0.6,'hue': 0.4}, 
		cjitter_p = 1, 
		randaffine = {'degrees': [-10,10], 'translate': [0.2, 0.2], 'scale': [1.3, 1.4], 'shear': 1}, 
		randpersp = {'distortion_scale': 0.1, 'p': 0.2}, 
		gray_p = 0.2, 
		gaussian_blur = {'kernel_size': 3, 'sigma': [0.1, 0.5]},
		rand_eras = {'p': 0.5, 'scale': [0.02, 0.33], 'ratio': [0.3, 3.3], 'value': 0}, 
		invert_p = 0.05,
		gaussian_noise = {'mean': 0., 'std': 0.004},
		gn_p = 0.0,
		n_test_crops = 10):

		randaffine['interpolation'] = randpersp['interpolation'] = T.InterpolationMode.BILINEAR
		randaffine['fill'] = randpersp['fill'] = [255, 255, 255]

		train_transforms = T.Compose([
			T.RandomCrop(size = img_crop_size, padding = None, pad_if_needed = True, fill = (255, 255, 255), padding_mode = 'constant'),
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

		val_transforms = T.Compose([
			T.RandomCrop(size = img_crop_size, padding = None, pad_if_needed = True, fill = (255, 255, 255), padding_mode = 'constant'),
			T.ToTensor(),
			T.Normalize(self.mean, self.std)
			])

		test_transforms = T.Compose([
            NRandomCrop(size = img_crop_size, n = n_test_crops, pad_if_needed = True),
            T.Lambda(lambda crops: torch.stack([T.Normalize(self.mean, self.std)(T.ToTensor()(crop)) for crop in crops]))
            ])
		
		if self.phase == 'train':
			return train_transforms			
		elif self.phase == 'val':
			return val_transforms
		elif self.phase == 'test':
			return test_transforms

	def generate_dataset(self):
		return datasets.ImageFolder(root = self.directory, 
			transform = self.compose_transform())

	def load_data(self):
		dataset = self.generate_dataset()
		if self.phase == 'train':
			if self.weighted_sampling:
				weights = self.make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes))
				weights = torch.DoubleTensor(weights)
				sampler = WeightedRandomSampler(weights, len(weights))
				loader = DataLoader(dataset, batch_size = self.batch_size, sampler = sampler)
			else:
				loader = DataLoader(dataset, batch_size = self.batch_size, shuffle = self.shuffle)
		else:
			loader = DataLoader(dataset, batch_size = self.batch_size, shuffle = self.shuffle)
		return dataset, loader

# Training

class save_results():
    def __init__(self, history_path, checkpoint_path, test_name):
        self.history_path = history_path
        self.checkpoint_path = checkpoint_path
        self.test_name = test_name

    def save_pkl(self, name, list_):
        with open(self.history_path + os.sep + self.test_name + name +'.pkl', 'wb') as f:
            pkl.dump(list_, f) 

    def save_checkpoints(self, ep_loss, min_loss, model, optimizer, name):
        if ep_loss <= min_loss:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': ep_loss}, self.checkpoint_path + os.sep + self.test_name + name + '_best_model.pth')
            return ep_loss
        else:
            return min_loss

def set_optimizer(optim_type, lr_, model):
    if optim_type == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
            lr = lr_,
            betas = [0.9, 0.999],
            weight_decay = 0.0001)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
            lr = lr_,
            momentum = 0.9,
            nesterov = False,
            weight_decay = 0.0001)
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
            lr = lr_,
            betas = [0.9, 0.999],
            weight_decay = 0.0001)
    
    ### AdamW è stato aggiunto successivamente -> speed-up del Training e riduzione dell'Overfitting
    elif optim_type == 'adamw':
          optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
            lr = lr_,
            betas = [0.9, 0.999],
            weight_decay = 0.0001)
    else:
        raise Exception('The selected optimization type is not available.')

    return optimizer

class Trainer():
    def __init__(self, model, t_set, v_set, DEVICE, optim_type, lr_, model_path, history_path, test_ID, num_epochs = 10):
        self.model = model
        self.t_set = t_set
        self.v_set = v_set
        self.DEVICE = DEVICE
        self.optim_type = optim_type
        self.lr_ = lr_
        self.model_path = model_path
        self.history_path = history_path
        self.test_ID = test_ID
        self.test_name = 'Test_' + self.test_ID + '_MLC_'
        self.num_epochs = num_epochs
        self.checkpoint_path = os.path.join(self.model_path, 'checkpoints')

    def compute_minibatch_accuracy(self, output, label):
        max_index = output.max(dim = 1)[1]
        return (max_index == label).sum().cpu().item(), (max_index == label).sum().cpu().item()/label.size()[0] 

    def train_model(self):

        optimizer = set_optimizer(self.optim_type, self.lr_, self.model)

        criterion = nn.CrossEntropyLoss()

        sr = save_results(self.history_path, self.checkpoint_path, self.test_name)

        self.model.train()
        
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []        
        min_loss_t = sys.maxsize
        min_loss_v = sys.maxsize                
        
        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch} / {self.num_epochs}')
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            for data, target in tqdm(self.t_set, 'Training'):
                data = data.to(self.DEVICE)
                target = target.to(self.DEVICE)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output,target)
                true, _ = self.compute_minibatch_accuracy(output, target)
                epoch_loss += float(loss.item())*len(data)
                epoch_acc += true
                loss.backward()
                optimizer.step()

            epoch_loss /= len(self.t_set.dataset)
            epoch_acc /= len(self.t_set.dataset)
            print(f'train_loss: {epoch_loss} - train_accuracy: {epoch_acc}')
            print()
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)

            min_loss_t = sr.save_checkpoints(epoch_loss, min_loss_t, self.model, optimizer, 'train')

            epoch_val_loss = 0.0
            epoch_val_acc = 0.0
            optimizer.zero_grad()
            self.model.eval()
            for data, target in tqdm(self.v_set, 'Validation'):
                data = data.to(self.DEVICE)
                target = target.to(self.DEVICE)
                with torch.no_grad():
                    output_val = self.model(data)                    
                validation_loss = criterion(output_val,target)
                val_true, _ = self.compute_minibatch_accuracy(output_val, target)                    
                epoch_val_loss += validation_loss.item()*len(data)
                epoch_val_acc += val_true
            
            epoch_val_loss /= len(self.v_set.dataset)
            epoch_val_acc /= len(self.v_set.dataset)
            print(f'val_loss: {epoch_val_loss} - val_accuracy: {epoch_val_acc}')
            print()
            val_loss.append(epoch_val_loss)
            val_acc.append(epoch_val_acc)
            
            min_loss_v = sr.save_checkpoints(epoch_val_loss, min_loss_v, self.model, optimizer, 'val')

            sr.save_pkl('train_losses', train_loss)
            sr.save_pkl('val_losses', val_loss)
            sr.save_pkl('train_accuracy', train_acc)
            sr.save_pkl('val_accuracy', val_acc)

    def __call__(self):
        self.train_model()

# Evaluation

def produce_classification_reports(dl, device, model,output_dir, test_id):    
    dataset = dl.generate_dataset()
    _, set_ = dl.load_data()
    
    labels = []
    preds = []
    target_names = list(dataset.class_to_idx.keys())
    c_to_idx = dataset.class_to_idx
    idx_to_c = {c_to_idx[k]: k for k in list(c_to_idx.keys())}
    for data, target in set_:
        data = data.to(device)
        labels += list(target.numpy())
        target = target.to(device)
        
        bs, ncrops, c, h, w = data.size()
        
        with torch.no_grad():
            output = model(data.view(-1, c, h, w))
            max_index = output.max(dim = 1)[1]
            max_index = max_index.cpu().detach().numpy()
            max_index_over_10_crops = max_index.reshape(bs,ncrops)
            final_max_index = []
            for i in range(bs):
                final_max_index.append(np.argmax(np.bincount(max_index_over_10_crops[i,:])))
                
            preds += list(final_max_index)
    
    label_class_names = [idx_to_c[id_] for id_ in labels]
    pred_class_names = [idx_to_c[id_] for id_ in preds]
    
    cm = confusion_matrix(label_class_names, pred_class_names, labels=target_names)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    
    cmap = plt.get_cmap('Blues')
    
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max() / 1.5
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy = {:0.4f}; misclass = {:0.4f}'.format(accuracy, misclass))
    plt.savefig(f'{output_dir}Test_{test_id}_confusion_matrix_test.png')
    
    with open(f'{output_dir}Test_{test_id}_classification-report_test.txt', 'w') as f:
        f.write(classification_report(labels, preds, target_names=target_names))