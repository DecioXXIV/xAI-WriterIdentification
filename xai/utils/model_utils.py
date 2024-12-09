import torch, pickle
import torch.nn as nn
import torchvision.models as models
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

### ###### ###
### MODELS ###
### ###### ###
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

class NN_Classifier(nn.Module):
    def __init__(self, mode = None, cp_path = './', num_classes = 8):
        super().__init__()
        self.base_model = load_encoder(mode, cp_path)
        self.fc_layers = nn.Sequential()
        self.num_classes = num_classes
        new_layer = add_fc_layer('hidden', 1024,  32)
        class_layer = add_fc_layer('last', 32, self.num_classes)
        self.fc_layers.add_module('fc0', new_layer)
        self.fc_layers.add_module('fc1', class_layer)

    def forward(self, x):
        x = self.base_model(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.fc_layers(x)
        return x

class SVM_Classifier(nn.Module):
    def __init__(self, mode='frozen', cp_path='./'):
        super().__init__()
        self.base_model = load_encoder(mode, cp_path)
        self.tail = SVC(C=1.0, kernel='rbf', decision_function_shape='ovo', probability=True, random_state=24)
    
    def extract_features(self, input_dl, device):
        self.base_model.eval()
        features, labels = list(), list()
    
        with torch.no_grad():
            for data, target in input_dl:
                data, target = data.to(device), target.to(device)
                output = self.base_model(data)
                features.append(output)
                labels.append(target)
        
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        return features, labels
    
    def forward(self, x):
        x_encoded = self.base_model(x)
        if len(x_encoded.shape) == 1:
            x_encoded = x_encoded.unsqueeze(0)

        x_encoded = x_encoded.cpu().detach().numpy()
        x_out = self.tail.predict_proba(x_encoded)

        return torch.tensor(x_out, dtype=torch.float32)

class GB_Classifier(nn.Module):
    def __init__(self, mode='frozen', cp_path='./'):
        super().__init__()
        self.base_model = load_encoder(mode, cp_path)
        self.tail = GradientBoostingClassifier(max_depth=3)
    
    def extract_features(self, input_dl, device):
        self.base_model.eval()
        features, labels = list(), list()

        with torch.no_grad():
            for data, target in input_dl:
                data, target = data.to(device), target.to(device)
                output = self.base_model(data)
                features.append(output)
                labels.append(target)
        
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
    
        return features, labels
        
    def forward(self, x):
        x_encoded = self.base_model(x)
        if len(x_encoded.shape) == 1:
            x_encoded = x_encoded.unsqueeze(0)

        x_encoded = x_encoded.cpu().detach().numpy()
        x_out = self.tail.predict_proba(x_encoded)

        return torch.tensor(x_out, dtype=torch.float32)

### ##### ###
### UTILS ###
### ##### ###
def load_encoder(mode, cp_path):
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

def load_tail(model_tail_path):
    with open(model_tail_path, 'rb') as f:
        return pickle.load(f)