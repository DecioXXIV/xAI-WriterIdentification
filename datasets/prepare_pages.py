import os
from argparse import ArgumentParser
from datetime import datetime

from utils import load_metadata, save_metadata, get_page_dimensions

from datasets.utils import produce_pages, copy_not_masked_test_instances

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True, help="ID for the new test")
    return parser.parse_args()

### #### ###
### MAIN ###
### #### ###            
if __name__ == '__main__':
    args = get_args()
    TEST_ID = args.test_id
    EXP_METADATA_PATH = os.path.join(LOG_ROOT, f"{TEST_ID}-metadata.json")
    
    try: EXP_METADATA = load_metadata(EXP_METADATA_PATH)
    except Exception as e:
        print(e)
        exit(1)
    
    if "DATA_PREP_TIMESTAMP" in EXP_METADATA:
        print(f"*** IN RELATION TO THE EXPERIMENT '{TEST_ID}', THE INSTANCES HAVE ALREADY BEEN PREPARED! ***\n")
        exit(1)
    
    print("*** BEGINNING OF DATA PREPARATION PROCESS ***")
    
    BASE_ID, RET_ID = None, None
    try: BASE_ID, RET_ID = TEST_ID.split(':')
    except: BASE_ID = TEST_ID
    
    DATASET = EXP_METADATA["DATASET"]
    CLASSES_DATA = EXP_METADATA["CLASSES"]
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]

    FINAL_WIDTH, FINAL_HEIGHT = get_page_dimensions(DATASET)
    
    os.makedirs(f"{DATASET_ROOT}/{DATASET}/processed", exist_ok = True)
    
    # If the current experiment is "ret", retrieve the original 'test' instances first
    if RET_ID is not None:
        print(f"'{TEST_ID}' is a 'ret' experiment: Training (masked) instances are already in the folder, ready to be prepared; Test (not-masked) instances need to be retrieved!")
        for c, c_type in CLASSES_DATA.items(): copy_not_masked_test_instances(c, c_type, DATASET, TEST_ID, MODEL_TYPE)
    
    # Process and extract sub-images for each class
    for c, c_type in CLASSES_DATA.items():
        produce_pages(c, c_type, DATASET, BASE_ID, MODEL_TYPE, FINAL_WIDTH, FINAL_HEIGHT)
    
    EXP_METADATA["DATA_PREP_TIMESTAMP"] = str(datetime.now())
    save_metadata(EXP_METADATA, EXP_METADATA_PATH)
    
    print("*** END OF DATA PREPARATION PROCESS ***\n")