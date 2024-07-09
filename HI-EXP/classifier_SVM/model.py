import torch, pickle
import torch.nn as nn
import torchvision.models as models
from sklearn.svm import SVC

class BaseModel(nn.Module):
    def __init__(self,):
        super().__init__()
        encoder = models.__dict__['resnet18'](num_classes = 512)
        encoder = nn.Sequential(*(list(encoder.children())[:-1]))
        self.enc = encoder

    def forward(self, x):
        x = self.enc(x).squeeze()
        return x

class SVM_Classifier(nn.Module):
    def __init__(self, mode='frozen', cp_path='./', num_classes=8):
        super().__init__()
        self.base_model = self.load_encoder(mode, cp_path)
        self.tail = SVC(C=1.0, kernel='rbf', decision_function_shape='ovo', random_state=24)
    
    def extract_features(self, input_dl):
        self.base_model.eval()
        features, labels = list(), list()

        with torch.no_grad():
            for data, target in input_dl:
                output = self.base_model(data)
                features.append(output)
                labels.append(target)
        
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
    
        return features, labels
    
    def load_encoder(self, mode, cp_path):
        cp = torch.load(cp_path)['model_state_dict']
        cp.pop('alpha')
        cp.pop('fc_layers.fc0.0.weight')
        cp.pop('fc_layers.fc0.0.bias')
        base_model = BaseModel()

        base_model.load_state_dict(cp)
   
        if mode == 'frozen':
            for param in base_model.parameters():
                param.requires_grad = False
    
        return base_model
    
    def load_tail(self, svm_model_path):
        with open(svm_model_path, 'rb') as f:
            self.tail = pickle.load(f)
    
    def forward(self, x):
        x_encoded, _ = self.extract_features(x)
        x_encoded = x_encoded.detatch().numpy()
        x_out = self.tail.predict_proba(x_encoded)

        return torch.tensor(x_out)