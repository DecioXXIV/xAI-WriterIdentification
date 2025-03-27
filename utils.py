import json, os
from argparse import ArgumentTypeError

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

def xaiaug2str(value):
    if value.lower() in ("pi", "protect and inform", "protect_inform"): return "pi"
    elif value.lower() in ("lr", "lime reds", "lime_reds"): return "lr"
    elif value.lower() in ("wo", "world opening", "world_opening"): return "wo"
    elif value.lower() in ("rand", "random"): return "rand"
    else: raise ArgumentTypeError("Unrecognized XAI Augmentation Mode")

def datasets2hi(value):
    valid_datasets = os.listdir(DATASETS_ROOT)
    valid_datasets.remove("__pycache__")
    valid_datasets.remove("__init__.py")
    valid_datasets.remove("prepare_pages.py")
    valid_datasets.remove("utils.py")
    
    if value in valid_datasets: return value
    else: raise ArgumentTypeError("Unrecognized Dataset")

### ################# ###
### METADATA HANDLING ###
### ################# ###
def load_metadata(metadata_path: str) -> dict:
    try:
        with open(metadata_path, 'r') as jf: return json.load(jf)
    except json.JSONDecodeError as e: 
        raise ValueError(f"Error occurred while decoding JSON file '{metadata_path}': {e}")
    except Exception as e: 
        raise FileNotFoundError(f"Error occurred while reading metadata file '{metadata_path}': {e}")

def save_metadata(metadata: dict, metadata_path: str):
    with open(metadata_path, 'w') as jf:
        json.dump(metadata, jf, indent=4)

### ################ ###
### DATASET HANDLING ###
### ################ ###
def get_train_instance_patterns():
    train_instance_patterns = {
        "CEDAR_Letter": lambda f: 'c' not in f,
        "CVL": lambda f: "-3" not in f and "-7" not in f,
        "VatLat653": lambda f: 'a' in f,
        "VatLat5951b": lambda f: 'a' in f
    }
    
    return train_instance_patterns

def get_test_instance_patterns():
    test_instance_patterns = {
        "CEDAR_Letter": lambda f: 'c' in f,
        "CVL": lambda f: "-3" in f or "-7" in f,
        "VatLat653": lambda f: 't' in f,
        "VatLat5951b": lambda f: 't' in f
    }
    
    return test_instance_patterns

def get_vert_hor_cuts(dataset):
    vert_cuts, hor_cuts = None, None
    
    if dataset == "CEDAR_Letter": vert_cuts, hor_cuts = 7, 4
    if dataset == "CVL": vert_cuts, hor_cuts = 5, 2
    if dataset == "VatLat653": vert_cuts, hor_cuts = 1, 1
    if dataset == "VatLat5951b": vert_cuts, hor_cuts = 1, 1
    if dataset == "VatLat4220_4221a": vert_cuts, hor_cuts = 1, 1
    
    return vert_cuts, hor_cuts

def get_page_dimensions(dataset):
    final_width, final_height = None, None
    
    if dataset == "CEDAR_Letter": final_width, final_height = 902, 1279
    if dataset == "CVL": final_width, final_height = 902, 1279
    if dataset == "VatLat653": final_width, final_height = 902, 1279
    if dataset == "VatLat5951b": final_width, final_height = 682, 1043
    if dataset == "VatLat4220_4221a": final_width, final_height = 1286, 2037
    
    return final_width, final_height

### ######################### ###
### MODEL CHECKPOINT HANDLING ###
### ######################### ###
def get_model_base_checkpoint(model_type):
    if model_type == "ResNet18":
        return f"{CLASSIFIERS_ROOT}/classifier_{model_type}/cp/Test_3_TL_val_best_model.pth"
    