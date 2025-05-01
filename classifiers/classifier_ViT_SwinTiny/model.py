import torch
import torch.nn as nn
import torchvision.models as models
 
CLASSIFIERS_ROOT = "./classifiers"
 
### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def add_fc_layer(type_fc_layer, in_f, out_f):
    if type_fc_layer == "last":
        fc = nn.Linear(in_features=in_f, out_features=out_f)
        nn.init.xavier_normal_(fc.weight)
        if fc.bias is not None: nn.init.zeros_(fc.bias)
         
        fc_block = nn.Sequential(fc)
     
    elif type_fc_layer == "hidden":
        components = list()
 
        fc = nn.Linear(in_features=in_f, out_features=out_f)
        nn.init.xavier_normal_(fc.weight)
        if fc.bias is not None: nn.init.zeros_(fc.bias)
        components.append(fc)
 
        components.append(nn.BatchNorm1d(out_f))
        components.append(nn.GELU())
 
        fc_block = nn.Sequential(*components)
     
    else: raise Exception('Wrong fully connected layer type')
 
    return fc_block

def load_vit_swintiny_classifier(num_classes, phase, test_id, exp_metadata, device, logger):
    model = ViT_SwinTiny_Classifier(num_classes=num_classes)
    output_dir = f"{CLASSIFIERS_ROOT}/classifier_ViT_SwinTiny/tests/{test_id}/output"
    last_cp = None
     
    if phase == "train":
        logger.info("'train' Phase")
        if "refined" in test_id:
            base_id, _ = test_id.split(':')
            last_cp_path = f"{CLASSIFIERS_ROOT}/classifier_ViT_SwinTiny/tests/{base_id}/output/checkpoints/Test_{base_id}_MLC_val_best_model.pth"
            last_cp = torch.load(last_cp_path)
            model.load_state_dict(last_cp['model_state_dict'])
             
            logger.warning(f"'{test_id}' is a 'refined' experiment: the Fine-Tuning will start from the model already fine-tuned for '{base_id}'")
         
        if "EPOCHS_COMPLETED" in exp_metadata:
            epochs = exp_metadata["FINE_TUNING_HP"]["total_epochs"]
            epochs_completed = exp_metadata["EPOCHS_COMPLETED"]
            epochs_to_do = epochs - epochs_completed
            last_cp_path = f"{output_dir}/checkpoints/Test_{test_id}_MLC_last_checkpoint.pth"
            last_cp = torch.load(last_cp_path)
            model.load_state_dict(last_cp['model_state_dict'])
 
            logger.warning(f"Fine-Tuning process for '{test_id}' has been somehow interrupted before its ending")
            logger.warning(f"{epochs_completed} epochs have already been completed: the Fine-Tuning process will be ended with the remaining {epochs_to_do} epochs")
     
    elif phase == "test":
        cp_to_test = f"{output_dir}/checkpoints/Test_{test_id}_MLC_val_best_model.pth"
        model.load_state_dict(torch.load(cp_to_test)['model_state_dict'])
        model.eval()
        logger.info("'test' Phase: the Model has been set-up for Testing phase")
     
    return model.to(device), last_cp
 
### ###### ###
### MODELS ###
### ###### ###
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        pre_trained = models.swin_t(weights = models.Swin_T_Weights.IMAGENET1K_V1)
        pre_trained = nn.Sequential(*(list(pre_trained.children())[:-1]))
        self.enc = pre_trained
 
    def forward(self, x):
        x = self.enc(x).squeeze()
        return x
 
class ViT_SwinTiny_Classifier(nn.Module):
    def __init__(self, num_classes = 8):
        super().__init__()
        self.base_model = BaseModel()
        self.num_classes = num_classes
        self.fc_layers = nn.Sequential()
        
        self.fc_layers.add_module("ch0", add_fc_layer("hidden", 768, 128))
        self.fc_layers.add_module("ch1", add_fc_layer("last", 128, self.num_classes))
 
    def extract_vis_features(self, x):
        x = self.base_model.enc(x)
        return x.squeeze()
        
    def forward(self, x):
        x = self.base_model(x)
        if x.dim() == 1: x = x.unsqueeze(0)
        x = self.fc_layers(x)
        return x