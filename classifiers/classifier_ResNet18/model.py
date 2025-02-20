import torch
import torch.nn as nn
import torchvision.models as models

CLASSIFIERS_ROOT = "./classifiers"

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def add_fc_layer(type_fc_layer, in_f, out_f):
    if type_fc_layer == 'last':
        fc_block = nn.Sequential(*[nn.Linear(in_features = in_f, out_features = out_f)])
    elif type_fc_layer == 'hidden':
        fc_block = nn.Sequential(*[nn.Linear(in_features = in_f, out_features = out_f),
                    nn.BatchNorm1d(out_f), nn.ReLU(), nn.Dropout(p = 0.3)])
    else:
        raise Exception('Wrong fully connected layer type')
    
    return fc_block

def load_encoder(mode, cp_path):
    cp = torch.load(cp_path)['model_state_dict']
    cp.pop('alpha')

    base_model = BaseModel()

    base_model.load_state_dict(cp)
   
    if mode == 'frozen':
        for param in base_model.parameters():
            param.requires_grad = False
    
    return base_model

def load_resnet18_classifier(num_classes, mode, cp_base, phase, test_id, exp_metadata, device):
    model = ResNet18_Classifier(num_classes=num_classes, mode=mode, cp_path=cp_base)
    output_dir = f"{CLASSIFIERS_ROOT}/classifier_ResNet18/tests/{test_id}/output"
    last_cp = None
    
    if phase == "train":
        if "refine" in test_id:
            base_id, _ = test_id.split(':')
            last_cp_path = f"{CLASSIFIERS_ROOT}/classifier_ResNet18/tests/{base_id}/output/checkpoints/Test_{base_id}_MLC_val_best_model.pth"
            last_cp = torch.load(last_cp_path)
            model.load_state_dict(last_cp['model_state_dict'])
        
        if "EPOCHS_COMPLETED" in exp_metadata:
            epochs = exp_metadata["FINE_TUNING_HP"]["total_epochs"]
            epochs_completed = exp_metadata["EPOCHS_COMPLETED"]
            epochs_to_do = epochs - epochs_completed
            print(f"{epochs_completed} epochs have already been completed: the Fine-Tuning process will be ended with the remaining {epochs_to_do} epochs")
            last_cp_path = f"{output_dir}/checkpoints/Test_{test_id}_MLC_last_checkpoint.pth"
            last_cp = torch.load(last_cp_path)
            model.load_state_dict(last_cp['model_state_dict'])
    
    if phase == "test":
        cp_to_test = f"{output_dir}/checkpoints/Test_{test_id}_MLC_val_best_model.pth"
        model.load_state_dict(torch.load(cp_to_test)['model_state_dict'])
        model.eval()
    
    return model.to(device), last_cp

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

class ResNet18_Classifier(nn.Module):
    def __init__(self, mode = None, cp_path = './', num_classes = 8):
        super().__init__()
        self.base_model = load_encoder(mode, cp_path)
        self.fc_layers = nn.Sequential()
        self.num_classes = num_classes
        new_layer = add_fc_layer('hidden', 1024,  32)
        class_layer = add_fc_layer('last', 32, self.num_classes)
        self.fc_layers.add_module('fc0', new_layer)
        self.fc_layers.add_module('fc1', class_layer)
    
    def extract_visual_features(self, x):
        x = self.base_model.enc(x)
        return x.squeeze()

    def forward(self, x):
        x = self.base_model(x)
        if x.dim() == 1: x = x.unsqueeze(0)
        x = self.fc_layers(x)
        return x