import os, json
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True, help="ID for the new test")
    return parser.parse_args()

def load_metadata(metadata_path) -> dict:
    try:
        with open(metadata_path, 'r') as jf: return json.load(jf)
    except json.JSONDecodeError as e: raise ValueError(f"Error occurred while decoding JSON file '{metadata_path}': {e}")
    except Exception as e: raise FileNotFoundError(f"Error occurred while reading metadata file '{metadata_path}': {e}")

def process_images(class_name, class_type, dataset, test_id, model_type, final_width, final_height, vert_mult, hor_mult):
    # Set 'source' and 'destination' for the current class, basing on its 'class_type'
    class_source, class_dest = None, None
    if class_type == "base": 
        class_source = f"{DATASET_ROOT}/{dataset}/{class_name}"
        class_dest = f"{DATASET_ROOT}/{dataset}/processed/{class_name}"
    else: 
        class_source = f"{DATASET_ROOT}/{dataset}/{class_name}-{test_id}_{model_type}_{class_type}"
        class_dest = f"{DATASET_ROOT}/{dataset}/processed/{class_name}-{test_id}_{model_type}_{class_type}"
    os.makedirs(class_dest, exist_ok=True)
    
    instances = os.listdir(class_source)
    
    for file in tqdm(instances, desc=f"Preparing Images of class {class_name}"):
        img = Image.open(f"{class_source}/{file}")
        img_width, img_height = img.size
        
        """Calculate the number of cuts, basing on...
        - Current image dimensions
        - Final dimensions (width=902, height=1279)
        - Multiplicative Factors: more high are the factors, more 902x1279 crops will be part of the training and test datasets
        """
        vert_cuts = (img_width // final_width) * vert_mult
        hor_cuts = (img_height // final_height) * hor_mult
        
        h_overlap = max(1, int((((vert_cuts + 1) * final_width) - img_width) / vert_cuts))
        v_overlap = max(1, int((((hor_cuts + 1) * final_height) - img_height) / hor_cuts))
        
        for h_cut in range(0, hor_cuts+1):
            for v_cut in range(0, vert_cuts+1):
                left = v_cut * (final_width - h_overlap)
                right = left + final_width
                top = h_cut * (final_height - v_overlap)
                bottom = top + final_height
                
                crop = img.crop((left, top, right, bottom))
                crop.save(f"{class_dest}/{file[:-4]}_{h_cut}_{v_cut}{file[-4:]}")
                
                # file[:-4] -> filename
                # file[-4:] -> filetype (eg: .jpg, .png, ...)

def copy_not_masked_test_instances(class_name, class_type, dataset, test_id, model_type):
    """This function is specific for the 'ret' Experiments.
    Its purpose is to include the Test Instances (selected with specific
    rules depending on the employed Dataset) in the Data Preparation process"""
    
    # In the "ret" Experiments, the 'test_id' is composed by two parts
    # base_id -> Test Name (eg: CEDAR-Letter-1)
    # ret_id -> Re-Training specifications (eg: ret0.1_saliency_all)
    base_id, _ = test_id.split(':')
    
    class_source = f"{DATASET_ROOT}/{dataset}/{class_name}" 
    class_dest = f"{DATASET_ROOT}/{dataset}/{class_name}-{base_id}_{model_type}_{class_type}"
    
    os.makedirs(class_dest, exist_ok=True)
    
    test_instances = None
    if dataset == "CEDAR_Letter": test_instances = [f for f in os.listdir(class_source) if "c" in f]
    if dataset == "CVL": test_instances = [f for f in os.listdir(class_source) if ("-3" in f or "-7" in f)]
    
    for f in test_instances: os.system(f"cp {class_source}/{f} {class_dest}/{f}")

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
    
    if "DATA_PREP_TIMESTAMP" in EXP_METADATA:
        print("*** IN RELATION TO THE SPECIFIED EXPERIMENT, THE INSTANCES HAVE ALREADY BEEN PREPARED! ***\n")
        exit(1)
    
    print("*** BEGINNING OF DATA PREPARATION PROCESS ***")
    
    BASE_ID, RET_ID = None, None
    try: BASE_ID, RET_ID = TEST_ID.split(':')
    except: BASE_ID = TEST_ID
    
    DATASET = EXP_METADATA["DATASET"]
    CLASSES_DATA = EXP_METADATA["CLASSES"]
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]

    FINAL_WIDTH, FINAL_HEIGHT = 902, 1279
    VERT_MULT_FACT = EXP_METADATA["PREP_MULT_FACT"]["VERT"]
    HOR_MULT_FACT = EXP_METADATA["PREP_MULT_FACT"]["HOR"]
    
    os.makedirs(f"{DATASET_ROOT}/{DATASET}/processed", exist_ok = True)
    
    if RET_ID is not None:
        for c, c_type in CLASSES_DATA.items(): copy_not_masked_test_instances(c, c_type, DATASET, TEST_ID, MODEL_TYPE)
    
    for c, c_type in CLASSES_DATA.items():
        process_images(c, c_type, DATASET, BASE_ID, MODEL_TYPE, FINAL_WIDTH, 
                       FINAL_HEIGHT, VERT_MULT_FACT, HOR_MULT_FACT)
    
    EXP_METADATA["DATA_PREP_TIMESTAMP"] = str(datetime.now())
    with open(f"{LOG_ROOT}/{TEST_ID}-metadata.json", 'w') as jf: json.dump(EXP_METADATA, jf, indent=4)
    
    print("***END OF DATA PREPARATION PROCESS ***\n")