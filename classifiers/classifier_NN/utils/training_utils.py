import torch, os, sys
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List

from utils import save_metadata

LOG_ROOT = "./log"

### ################## ###
### SAVINGS & LOADINGS ###
### ################## ###
def load_history(filepath) -> List[float]:
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pkl.load(f)
    
    else: return []

def save_history(filepath: str, data: List[float]):
    with open(filepath, "wb") as f:
        pkl.dump(data, f)
        
def save_checkpoint(epoch_loss, min_loss, model, optimizer, filepath, phase, check=True):
    # Saving the New Best Checkpoint: check "train_loss" and "val_loss"
    if check is True and epoch_loss <= min_loss:
        checkpoint_path = f"{filepath}{phase}_best_model.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss}, checkpoint_path)
        return epoch_loss

    # Saving the Last Checkpoint: this allows to restore the Training Process if interrupted before its ending
    elif check is False:
        checkpoint_path = f"{filepath}last_checkpoint.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss}, checkpoint_path)
    
    return min_loss if check else None

def plot_metric(metric, output_dir, test_id):
    values = {"train": [], "val": []}
    
    for phase in values.keys():
        with open(f'{output_dir}/Test_{test_id}_MLC_{phase}_{metric}.pkl', 'rb') as f:
            values[phase] = pkl.load(f)
    
    best_train_metric, best_val_metric = None, None
    if metric == "loss": best_train_metric, best_val_metric = np.min(values['train']), np.min(values['val'])
    if metric == "accuracy": best_train_metric, best_val_metric = np.max(values['train']), np.max(values['val'])
    best_train_epoch = np.where(np.array(values['train']) == best_train_metric)[0][0] + 1
    best_val_epoch = np.where(np.array(values['val']) == best_val_metric)[0][0] + 1
    
    with open(f'{output_dir}/Test_{test_id}_MLC_{metric}.txt', 'w') as f:
        f.write(f"The optimal value of {metric} for the training set is: {round(best_train_metric, 3)}\n")
        f.write(f"The optimal value of {metric} for the validation set is: {round(best_val_metric, 3)}\n")
        f.write(f"Epoch corresponding to the optimal value of the training {metric}: {best_train_epoch}\\{len(values['train'])}\n")
        f.write(f"Epoch corresponding to the optimal value of the validation {metric}: {best_val_epoch}\\{len(values['val'])}\n")
    
    plt.plot(values['train'])
    plt.plot(values['val'])
    plt.title(f"Model {metric}")
    plt.ylabel(f"{metric} [-]")
    plt.xlabel("Epoch [-]")
    plt.legend(['Training', 'Validation'], loc='best')
    plt.savefig(f'{output_dir}/Test_{test_id}_MLC_{metric}.png')
    plt.close()
            
### ####### ###
### TRAINER ###
### ####### ###
def set_optimizer(optim_type, lr_, model, cp=None):
    if optim_type == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
            lr = lr_, betas = [0.9, 0.999], weight_decay = 0.0001)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
            lr = lr_, momentum = 0.9, nesterov = False, weight_decay = 0.0001)
    elif optim_type == 'nesterov':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
            lr = lr_, momentum = 0.9, nesterov = True, weight_decay = 0.0001)
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
            lr = lr_, betas = [0.9, 0.999], weight_decay = 0.0001)
    else:
        raise Exception('The selected optimization type is not available.')
    
    if cp is not None: optimizer.load_state_dict(cp["optimizer_state_dict"])

    return optimizer

class Trainer():
    def __init__(self, model, t_set, v_set, DEVICE, model_path, history_path, exp_metadata, last_cp=None):
        self.model = model
        self.t_set = t_set
        self.v_set = v_set
        self.device = DEVICE
        self.model_path = model_path
        self.history_path = history_path
        self.checkpoint_path = os.path.join(self.model_path, 'checkpoints')
        self.exp_metadata = exp_metadata
        self.last_cp = last_cp
        
        self.optim_type = self.exp_metadata["FINE_TUNING_HP"]["optimizer"]
        self.lr_ = self.exp_metadata["FINE_TUNING_HP"]["learning_rate"]
        self.num_epochs = self.exp_metadata["FINE_TUNING_HP"]["total_epochs"]
        
        self.test_id = self.exp_metadata["TEST_ID"]
        self.test_name = f"Test_{self.test_id}_MLC_"
    
    def compute_minibatch_accuracy(self, output: torch.Tensor, label: torch.Tensor) -> float:
        max_index = output.argmax(dim=1)
        correct = (max_index == label).sum().item()
        correct_ratio = correct / label.size(0)
        
        return correct, correct_ratio

    def train_one_epoch(self, optimizer, criterion):
        self.model.train()
        epoch_loss, epoch_acc = 0.0, 0.0
        
        for data, target in tqdm(self.t_set, desc="Training"):
            bs, ncrops, c, h, w = data.size()
            target_expanded = target.repeat_interleave(ncrops)
            
            data = data.view(-1, c, h, w).to(self.device)
            target_expanded = target_expanded.to(self.device)
            
            # Shuffle data
            perm = torch.randperm(bs * ncrops)
            data, target_expanded = data[perm], target_expanded[perm]
            
            # Training Step
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target_expanded)
            correct, _ = self.compute_minibatch_accuracy(output, target_expanded)
            
            epoch_loss += loss.item() * bs
            epoch_acc += correct
            
            loss.backward()
            optimizer.step()
        
        epoch_final_loss = epoch_loss / len(self.t_set.dataset)
        epoch_final_acc = epoch_acc / (len(self.t_set.dataset) * ncrops)
        
        print(f"Train_Loss: {epoch_final_loss} - Train_Accuracy: {epoch_final_acc}\n")
        
        return epoch_final_loss, epoch_final_acc

    def validate_one_epoch(self, criterion):
        self.model.eval()
        val_loss, val_acc = 0.0, 0.0
        
        with torch.no_grad():
            for data, target in tqdm(self.v_set, desc="Validation"):
                bs, ncrops, c, h, w = data.size()
                target_expanded = target.repeat_interleave(ncrops)
                
                data = data.view(-1, c, h, w).to(self.device)
                target_expanded = target_expanded.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target_expanded)
                correct, _ = self.compute_minibatch_accuracy(output, target_expanded)
                
                val_loss += loss.item() * bs
                val_acc += correct
        
        epoch_final_loss = val_loss / len(self.v_set.dataset)
        epoch_final_acc = val_acc / (len(self.v_set.dataset) * ncrops)
        
        print(f"Val_Loss: {epoch_final_loss} - Val_Accuracy: {epoch_final_acc}\n")
        
        return epoch_final_loss, epoch_final_acc

    def train_model(self):
        optimizer = set_optimizer(self.optim_type, self.lr_, self.model, self.last_cp)
        criterion = nn.CrossEntropyLoss()
        
        train_loss = load_history(f"{self.checkpoint_path}/../{self.test_name}train_loss.pkl")
        val_loss = load_history(f"{self.checkpoint_path}/../{self.test_name}val_loss.pkl")
        train_acc = load_history(f"{self.checkpoint_path}/../{self.test_name}train_accuracy.pkl")
        val_acc = load_history(f"{self.checkpoint_path}/../{self.test_name}val_accuracy.pkl")
         
        min_loss_t = sys.maxsize
        min_loss_v = sys.maxsize
                
        start_epoch = self.exp_metadata.get("EPOCHS_COMPLETED", 0) + 1
        tosave_cp_path = f"{self.checkpoint_path}/{self.test_name}"
        
        for epoch in range(start_epoch, self.num_epochs + 1):
            print(f'Epoch {epoch} / {self.num_epochs}')
            
            train_epoch_loss, train_epoch_acc = self.train_one_epoch(optimizer, criterion)
            val_epoch_loss, val_epoch_acc = self.validate_one_epoch(criterion)
            
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)
            
            # Save checkpoints and Training stats
            min_loss_t = save_checkpoint(train_epoch_loss, min_loss_t, self.model, optimizer, tosave_cp_path, "train", check=True)
            min_loss_v = save_checkpoint(val_epoch_loss, min_loss_v, self.model, optimizer, tosave_cp_path, "val", check=True)
            _ = save_checkpoint(train_epoch_loss, None, self.model, optimizer, tosave_cp_path, None, check=False)
            
            save_history(f"{self.history_path}/{self.test_name}train_loss.pkl", train_loss)
            save_history(f"{self.history_path}/{self.test_name}val_loss.pkl", val_loss)
            save_history(f"{self.history_path}/{self.test_name}train_accuracy.pkl", train_acc)
            save_history(f"{self.history_path}/{self.test_name}val_accuracy.pkl", val_acc)
            
            # Update metadata
            self.exp_metadata["EPOCHS_COMPLETED"] = epoch
            exp_metadata_path = f"{LOG_ROOT}/{self.test_id}-metadata.json"
            save_metadata(self.exp_metadata, exp_metadata_path)

    def __call__(self):
        self.train_model()