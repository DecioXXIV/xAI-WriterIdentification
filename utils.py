import os, json, logging, torch
from rich.logging import RichHandler
from argparse import ArgumentTypeError
from torchvision import transforms as T
from torch.nn import functional as F

LOG_ROOT = "./log"
DATASETS_ROOT = "./datasets"
CLASSIFIERS_ROOT = "./classifiers"

### ################## ###
### PARAMETER HANDLING ###
### ################## ###
def str2bool(value):
    if isinstance(value, bool): return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise ArgumentTypeError("Boolean value expected (true/false).")

def xaiaug2str(value: str):
    if value.lower() in ("pi", "protect and inform", "protect_inform"): return "pi"
    elif value.lower() in ("lr", "lime reds", "lime_reds"): return "lr"
    elif value.lower() in ("wo", "world opening", "world_opening"): return "wo"
    elif value.lower() in ("rand", "random"): return "rand"
    else: raise ArgumentTypeError("Unrecognized XAI Augmentation Mode")

def datasets2hi(value: str):
    valid_datasets = os.listdir(DATASETS_ROOT)
    valid_datasets.remove("__pycache__")
    valid_datasets.remove("__init__.py")
    valid_datasets.remove("prepare_pages.py")
    valid_datasets.remove("dataset_utils.py")
    
    value = value.strip()
    if value in valid_datasets: return value
    else: raise ArgumentTypeError("Unrecognized Dataset")

### ################# ###
### METADATA HANDLING ###
### ################# ###
def load_metadata(metadata_path: str, logger: logging.Logger) -> dict:
    try:
        with open(metadata_path, 'r') as jf: return json.load(jf)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in file '{metadata_path}': {e}")
        raise ValueError(f"Error occurred while decoding JSON file '{metadata_path}': {e}")
    except Exception as e:
        logger.error(f"Error occurred while reading metadata file '{metadata_path}': {e}") 
        raise FileNotFoundError(f"Error occurred while reading metadata file '{metadata_path}': {e}")

def save_metadata(metadata: dict, metadata_path: str):
    with open(metadata_path, 'w') as jf:
        json.dump(metadata, jf, indent=4)

def get_logger(test_id: str):
    logger = logging.getLogger(f"logger_{test_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    filepath = f"./log/{test_id}.log"
    file_handler = logging.FileHandler(filepath, mode='a')
    file_handler.setFormatter(formatter)
    
    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    rich_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(rich_handler)
    
    return logger

### ################ ###
### DATASET HANDLING ###
### ################ ###
def get_train_instance_patterns():
    train_instance_patterns = {
        "CEDAR_Letter": lambda f: 'c' not in f,
        "CVL": lambda f: "-3" not in f and "-7" not in f,
        "VatLat653": lambda f: 'a' in f,
        "VatLat5951b": lambda f: 'a' in f,
        "VatLat4221": lambda f: 'a' in f
    }
    
    return train_instance_patterns

def get_test_instance_patterns():
    test_instance_patterns = {
        "CEDAR_Letter": lambda f: 'c' in f,
        "CVL": lambda f: "-3" in f or "-7" in f,
        "VatLat653": lambda f: 't' in f,
        "VatLat5951b": lambda f: 't' in f,
        "VatLat4221": lambda f: 't' in f
    }
    
    return test_instance_patterns

def get_vert_hor_cuts(dataset):
    vert_cuts, hor_cuts = None, None
    
    if dataset == "CEDAR_Letter": vert_cuts, hor_cuts = 7, 4
    elif dataset == "CVL": vert_cuts, hor_cuts = 5, 2
    elif dataset == "VatLat653": vert_cuts, hor_cuts = 1, 1
    elif dataset == "VatLat5951b": vert_cuts, hor_cuts = 1, 1
    elif dataset == "VatLat4221": vert_cuts, hor_cuts = 1, 1
    
    return vert_cuts, hor_cuts

def get_page_dimensions(dataset):
    final_width, final_height = None, None
    
    if dataset == "CEDAR_Letter": final_width, final_height = 902, 1279
    elif dataset == "CVL": final_width, final_height = 902, 1279
    elif dataset == "VatLat653": final_width, final_height = 902, 1279
    elif dataset == "VatLat5951b": final_width, final_height = 682, 1043
    elif dataset == "VatLat4221": final_width, final_height = 1270, 2020
    
    return final_width, final_height

### ######################### ###
### MODEL CHECKPOINT HANDLING ###
### ######################### ###
def get_model_base_checkpoint(model_type):
    if model_type == "ResNet18":
        return f"{CLASSIFIERS_ROOT}/classifier_{model_type}/cp/Test_3_TL_val_best_model.pth"

### ############################## ###
### BATCH PREDICT FOR EXPLANATIONS ###
### ############################## ###
def get_batch_predict_function(model_type):
    if model_type == "ResNet18":
        def batch_predict(model, inputs, mean, std):
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            t_transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            batch = torch.stack(tuple(t_transforms(i) for i in inputs), dim=0)

            model.to(device)
            batch = batch.to(device)

            with torch.no_grad():
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
            
            return probs.detach().cpu().numpy()
    
        return batch_predict