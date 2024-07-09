import torch
import torch.nn as nn
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self,):
        super().__init__()
        encoder = models.__dict__['resnet18'](num_classes = 512)
        encoder = nn.Sequential(*(list(encoder.children())[:-1]))
        self.enc = encoder
        self.fc_layers = nn.Sequential()
        new_layer = self.add_fc_layer('last', 512,  1024)
        self.fc_layers.add_module('fc0', new_layer)
    
    def add_fc_layer(self, type_fc_layer, in_f, out_f):
        if type_fc_layer == 'last':
            fc_block = nn.Sequential(*[nn.Linear(in_features = in_f, out_features = out_f)])
        elif type_fc_layer == 'hidden':
            fc_block = nn.Sequential(*[nn.Linear(in_features = in_f, out_features = out_f),
                    nn.BatchNorm1d(out_f), nn.ReLU(), nn.Dropout(p = 0.3)])
        else:
            raise Exception('Wrong fully connected layer type')
    
        return fc_block 

    def forward(self, x):
        x = self.enc(x).squeeze()
        x = self.fc_layers(x)
        return x

class NN_Classifier(nn.Module):
    def __init__(self, mode = None, cp_path = './', num_classes = 8):
        super().__init__()
        self.base_model = self.load_encoder(mode, cp_path)
        self.fc_layers = nn.Sequential()
        self.num_classes = num_classes
        new_layer = self.add_fc_layer('hidden', 1024,  32)
        class_layer = self.add_fc_layer('last', 32, self.num_classes)
        self.fc_layers.add_module('fc0', new_layer)
        self.fc_layers.add_module('fc1', class_layer)
    
    def load_encoder(self, mode, cp_path):
        cp = torch.load(cp_path)['model_state_dict']
        cp.pop('alpha')

        base_model = BaseModel()

        base_model.load_state_dict(cp)
   
        if mode == 'frozen':
            for param in base_model.parameters():
                param.requires_grad = False
    
        return base_model

    def add_fc_layer(self, type_fc_layer, in_f, out_f):
        if type_fc_layer == 'last':
            fc_block = nn.Sequential(*[nn.Linear(in_features = in_f, out_features = out_f)])
        elif type_fc_layer == 'hidden':
            fc_block = nn.Sequential(*[nn.Linear(in_features = in_f, out_features = out_f),
                    nn.BatchNorm1d(out_f), nn.ReLU(), nn.Dropout(p = 0.3)])
        else:
            raise Exception('Wrong fully connected layer type')
    
        return fc_block 

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc_layers(x)
        return x