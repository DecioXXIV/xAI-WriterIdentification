import torch, os, sys
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

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
        
def save_checkpoint(epoch_loss, min_loss, model, optimizer, scheduler, early_stopping, filepath, phase, check=True):
    state_dict = dict()
    state_dict["model_state_dict"] = model.state_dict()
    state_dict["optimizer_state_dict"] = optimizer.state_dict()
    state_dict["loss"] = epoch_loss
    if scheduler is not None: state_dict["scheduler_state_dict"] = scheduler.state_dict()
    if early_stopping is not None: state_dict["early_stopping"] = early_stopping
    
    ### CARE! Early Stopping in "save_checkpoint" has to be fixed! ###    
    
    # Saving the New Best Checkpoint: check "train_loss" and "val_loss"
    if check is True and epoch_loss <= min_loss:
        checkpoint_path = f"{filepath}{phase}_best_model.pth"
        torch.save(state_dict, checkpoint_path)
        return epoch_loss

    # Saving the Last Checkpoint: this allows to restore the Training Process if interrupted before its ending
    elif check is False:
        checkpoint_path = f"{filepath}last_checkpoint.pth"
        torch.save(state_dict, checkpoint_path)
    
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

def plot_learning_rates(output_dir, test_id):
    with open(f"{output_dir}/Test_{test_id}_MLC_learning_rates.pkl", "rb") as f:
        learning_rates = pkl.load(f)
    
    plt.plot(learning_rates)
    plt.title("Learning Rate Schedule")
    plt.ylabel("Learning Rate [-]")
    plt.xlabel("Epoch [-]")
    plt.savefig(f'{output_dir}/Test_{test_id}_MLC_learning_rates.png')
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
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
            lr = lr_, betas = [0.9, 0.999], weight_decay = 0.01)
    else:
        raise Exception('The selected optimization type is not available.')
    
    if cp is not None: optimizer.load_state_dict(cp["optimizer_state_dict"])

    return optimizer

def set_scheduler(optimizer, exp_metadata, cp=None):
    ft_params = exp_metadata["FINE_TUNING_HP"]
    scheduler, scheduler_type = None, ft_params['scheduler']

    max_epochs = ft_params['total_epochs']
    min_lr = ft_params['learning_rate'] * 0.125
    patience = int(ft_params['total_epochs'] * 0.1)

    last_epoch = exp_metadata.get("EPOCHS_COMPLETED", -1)
    
    if scheduler_type == 'cos_annealing':
        scheduler = CosineAnnealingLR(optimizer, max_epochs, min_lr, last_epoch)
    elif scheduler_type == "reduce_lr_on_plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience)
    
    if scheduler is not None and cp is not None: scheduler.load_state_dict(cp["scheduler_state_dict"])
    
    return scheduler

def reset_cos_annealing_scheduler(cos_scheduler, total_epochs, exp_metadata):
    ft_params = exp_metadata["FINE_TUNING_HP"]
    last_epoch = exp_metadata["EPOCHS_COMPLETED"]
    max_epochs = total_epochs - last_epoch
    min_lr = int(ft_params['learning_rate'] * 0.001)
    
    cos_scheduler.T_max = max_epochs
    cos_scheduler.eta_min = min_lr
    
    return cos_scheduler
    
class EarlyStopping():
    def __init__(self, logger, patience=10, delta=0.0, max_epochs=200):
        self.logger = logger
        self.patience = patience
        self.delta = delta
        self.max_epochs = max_epochs
        self.patience_counter = 0
    
    def set_best_val_loss(self, val_loss):
        self.best_val_loss = val_loss
    
    def step_before_trigger(self, val_loss):
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
    
    def step(self, val_loss):
        if val_loss < self.best_val_loss - self.delta:
            self.logger.info(f"Validation loss improved from {self.best_val_loss} to {val_loss}. Resetting patience...")
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            self.logger.warning(f"Validation loss did not improve. Patience counter: {self.patience_counter}/{self.patience}.")
            if self.patience_counter >= self.patience:
                self.logger.warning("Early stopping triggered.")
                return True

class Trainer():
    def __init__(self, model, t_set, v_set, DEVICE, model_path, history_path, exp_metadata, use_early_stopping=True, last_cp=None, logger=None):
        self.model = model
        self.t_set = t_set
        self.v_set = v_set
        self.device = DEVICE
        self.model_path = model_path
        self.history_path = history_path
        self.checkpoint_path = os.path.join(self.model_path, 'checkpoints')
        self.exp_metadata = exp_metadata
        self.use_early_stopping = use_early_stopping
        self.last_cp = last_cp
        self.logger = logger
        
        self.optim_type = self.exp_metadata["FINE_TUNING_HP"]["optimizer"]
        self.lr_ = self.exp_metadata["FINE_TUNING_HP"]["learning_rate"]
        self.scheduler = self.exp_metadata["FINE_TUNING_HP"]["scheduler"]
        self.num_epochs = self.exp_metadata["FINE_TUNING_HP"]["total_epochs"]
        
        self.test_id = self.exp_metadata["TEST_ID"]
        self.test_name = f"Test_{self.test_id}_MLC_"

        self.max_epochs = None
        if self.use_early_stopping:
            self.early_stopping = EarlyStopping(self.logger, patience=10, delta=0.0, max_epochs=200)
            self.max_epochs = self.early_stopping.max_epochs
        else: self.max_epochs = self.num_epochs
    
    def compute_minibatch_accuracy(self, output: torch.Tensor, label: torch.Tensor) -> float:
        max_index = output.argmax(dim=1)
        correct = (max_index == label).sum().item()
        correct_ratio = correct / label.size(0)
        
        return correct, correct_ratio

    def train_one_epoch(self, optimizer, criterion):
        self.model.train()
        epoch_loss, epoch_acc, total_samples = torch.tensor(0.0), 0.0, 0
        pbar = tqdm(self.t_set, desc="Training", dynamic_ncols=True)

        for data, target in pbar:
            bs = data.size()[0]
            data, target, epoch_loss = data.to(self.device), target.to(self.device), epoch_loss.to(self.device)
            
            # Training Step
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            correct, _ = self.compute_minibatch_accuracy(output, target)
            
            epoch_loss += loss.detach() * bs
            epoch_acc += correct
            total_samples += bs
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=epoch_loss.item()/total_samples, accuracy=epoch_acc/total_samples, refresh=True)
        
        epoch_final_loss = epoch_loss.item() / total_samples
        epoch_final_acc = epoch_acc / total_samples
        
        return epoch_final_loss, epoch_final_acc

    def validate_one_epoch(self, criterion):
        self.model.eval()
        val_loss, val_acc, total_samples = torch.tensor(0.0), 0.0, 0
        pbar = tqdm(self.v_set, desc="Validation", dynamic_ncols=True)

        with torch.no_grad():
            for data, target in pbar:
                bs = data.size()[0]
                data, target, val_loss = data.to(self.device), target.to(self.device), val_loss.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                correct, _ = self.compute_minibatch_accuracy(output, target)
                
                val_loss += loss.detach() * bs
                val_acc += correct
                total_samples += bs
                
                pbar.set_postfix(loss=val_loss.item()/total_samples, accuracy=val_acc/total_samples, refresh=True)
        
        epoch_final_loss = val_loss.item() / total_samples
        epoch_final_acc = val_acc / total_samples
        
        return epoch_final_loss, epoch_final_acc

    def train_model(self):
        optimizer = set_optimizer(self.optim_type, self.lr_, self.model, self.last_cp)
        scheduler = set_scheduler(optimizer, self.exp_metadata, self.last_cp)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        train_loss = load_history(f"{self.checkpoint_path}/../{self.test_name}train_loss.pkl")
        val_loss = load_history(f"{self.checkpoint_path}/../{self.test_name}val_loss.pkl")
        train_acc = load_history(f"{self.checkpoint_path}/../{self.test_name}train_accuracy.pkl")
        val_acc = load_history(f"{self.checkpoint_path}/../{self.test_name}val_accuracy.pkl")
        learning_rates = load_history(f"{self.checkpoint_path}/../{self.test_name}learning_rates.pkl")
         
        if train_loss == []: min_loss_t = sys.maxsize
        else: min_loss_t = np.min(train_loss)
        if val_loss == []: min_loss_v = sys.maxsize
        else: min_loss_v = np.min(val_loss)
                
        start_epoch = self.exp_metadata.get("EPOCHS_COMPLETED", 0) + 1
        tosave_cp_path = f"{self.checkpoint_path}/{self.test_name}"
        
        if self.use_early_stopping:
            self.logger.warning(f"'Early Stopping' is enabled. Max epochs: {self.max_epochs}")
            self.logger.warning(f"The check on 'val_loss' will be triggered starting from epoch: {self.num_epochs}")
            if start_epoch > 1:
                self.early_stopping = torch.load(self.last_cp)["early_stopping"]
            self.early_stopping.set_best_val_loss(min_loss_v)
        
        for epoch in range(start_epoch, self.max_epochs + 1):
            self.logger.info(f'Epoch {epoch} / {self.max_epochs}')
            
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            if scheduler is not None: self.logger.info(f"Current Learning Rate: {current_lr}")
            
            if isinstance(scheduler, CosineAnnealingLR) and epoch == self.num_epochs:
                scheduler = reset_cos_annealing_scheduler(scheduler, self.max_epochs, self.exp_metadata)

            train_epoch_loss, train_epoch_acc = self.train_one_epoch(optimizer, criterion)
            self.logger.info(f"Epoch: {epoch} -> Train Loss: {train_epoch_loss} - Train Accuracy: {train_epoch_acc}")
            val_epoch_loss, val_epoch_acc = self.validate_one_epoch(criterion)
            self.logger.info(f"Epoch: {epoch} -> Val_Loss: {val_epoch_loss} - Val_Accuracy: {val_epoch_acc}\n")
            
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)
            
            # Save checkpoints and Training stats
            min_loss_t = save_checkpoint(train_epoch_loss, min_loss_t, self.model, optimizer, scheduler, self.early_stopping, tosave_cp_path, "train", check=True)
            min_loss_v = save_checkpoint(val_epoch_loss, min_loss_v, self.model, optimizer, scheduler, self.early_stopping, tosave_cp_path, "val", check=True)
            _ = save_checkpoint(train_epoch_loss, None, self.model, optimizer, scheduler, self.early_stopping, tosave_cp_path, None, check=False)
            
            save_history(f"{self.history_path}/{self.test_name}train_loss.pkl", train_loss)
            save_history(f"{self.history_path}/{self.test_name}val_loss.pkl", val_loss)
            save_history(f"{self.history_path}/{self.test_name}train_accuracy.pkl", train_acc)
            save_history(f"{self.history_path}/{self.test_name}val_accuracy.pkl", val_acc)
            save_history(f"{self.history_path}/{self.test_name}learning_rates.pkl", learning_rates)
            
            if scheduler is not None: 
                if isinstance(scheduler, CosineAnnealingLR): scheduler.step()
                elif isinstance(scheduler, ReduceLROnPlateau): scheduler.step(val_epoch_loss)
            
            # Update metadata
            self.exp_metadata["EPOCHS_COMPLETED"] = epoch
            exp_metadata_path = f"{LOG_ROOT}/{self.test_id}-metadata.json"
            save_metadata(self.exp_metadata, exp_metadata_path)

            if self.use_early_stopping:
                if epoch <= self.num_epochs: 
                    _ = self.early_stopping.step_before_trigger(val_epoch_loss)
                else:
                    condition = self.early_stopping.step(val_epoch_loss)
                    if condition: break
                
    def __call__(self):
        self.train_model()