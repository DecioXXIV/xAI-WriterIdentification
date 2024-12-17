import os, json
from datetime import datetime
from argparse import ArgumentParser
from torchvision import transforms as T
from typing import Tuple

from .image_masker import ImageMasker

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
XAI_ROOT = "./xai"

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase"])
    parser.add_argument("-mask_rate", type=float, required=True)
    parser.add_argument("-mode", type=str, required=True, choices=["saliency", "random"])

    return parser.parse_args()

def load_metadata(metadata_path) -> dict:
    try:
        with open(metadata_path, 'r') as jf: return json.load(jf)
    except json.JSONDecodeError as e: raise ValueError(f"Error occurred while decoding JSON file '{metadata_path}': {e}")
    except Exception as e: raise FileNotFoundError(f"Error occurred while reading metadata file '{metadata_path}': {e}")

def setup_masking_process(dataset, classes) -> Tuple[list, list]:
    dataset_dir = f"{DATASET_ROOT}/{dataset}"
    instances, instance_full_paths = list(), list()
    
    for c in classes:
        if dataset == "CEDAR_Letter":
            class_instances = [inst for inst in os.listdir(f"{dataset_dir}/{c}") if "c" not in inst]
        if dataset == "CVL":
            class_instances = [inst for inst in os.listdir(f"{dataset_dir}/{c}") if ("-3" not in inst and "-7" not in inst)]
        
        for i in class_instances: 
            instances.append(i)
            instance_full_paths.append(f"{dataset_dir}/{c}/{i}")
    
    return instances, instance_full_paths
            
### #### ###
### MAIN ###
### #### ###
if __name__ == '__main__':
    args = get_args()
    TEST_ID = args.test_id
    EXP_METADATA_PATH = f"{LOG_ROOT}/{TEST_ID}-metadata.json"

    try: EXP_METADATA = load_metadata(EXP_METADATA_PATH)
    except Exception as e:
        print(e)
        exit(1)
    
    MASK_RATE = args.mask_rate
    MODE = args.mode
        
    XAI_ALGORITHM = args.xai_algorithm
    if f"{XAI_ALGORITHM}_METADATA" not in EXP_METADATA:
        print(f"*** IN RELATION TO THE '{TEST_ID}' EXPERIMENT, THE XAI ALGORITHM '{XAI_ALGORITHM}' HAS NOT BEEN EMPLOYED YET! ***\n")
        exit(1)
    
    if f"MASK_PROCESS_{MODE}_{MASK_RATE}_END_TIMESTAMP" in EXP_METADATA:
        print(f"*** IN RELATION TO ['{TEST_ID}' EXPERIMENT, '{XAI_ALGORITHM}' XAI ALGORITHM, '{MODE}' MODE, '{MASK_RATE}' MASK RATE], THE MASKING PROCESS HAS ALREADY BEEN PERFORMED! ***\n")
        exit(1)
    
    DATASET = EXP_METADATA["DATASET"]
    CLASSES = list(EXP_METADATA["CLASSES"].keys())
    EXP_DIR = EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["DIR_NAME"]
    BLOCK_WIDTH = EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["BLOCK_DIM"]["WIDTH"]
    BLOCK_HEIGHT = EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["BLOCK_DIM"]["HEIGHT"]
    
    instances, paths = setup_masking_process(DATASET, CLASSES)
    
    masker = ImageMasker(
        instances=instances,
        paths=paths,
        test_id=TEST_ID,
        exp_dir=EXP_DIR,
        mask_rate=MASK_RATE,
        mode=MODE,
        block_width=BLOCK_WIDTH,
        block_height=BLOCK_HEIGHT,
        xai_algorithm=XAI_ALGORITHM,
        exp_metadata=EXP_METADATA)
    
    masker()
    
    EXP_METADATA[f"MASK_PROCESS_{MODE}_{MASK_RATE}_END_TIMESTAMP"] = str(datetime.now())
    with open(EXP_METADATA_PATH, "w") as jf: json.dump(EXP_METADATA, jf, indent=4) 