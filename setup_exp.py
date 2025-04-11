import os, logging
from argparse import ArgumentParser

from utils import save_metadata, datasets2hi, get_logger

LOG_ROOT = "./log"

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def get_args():
    parser = ArgumentParser()
    
    # General parameters
    parser.add_argument("-test_id", type=str, required=True, help="ID for the new Experiment to be performed")
    parser.add_argument("-model_type", type=str, required=True, choices=["ResNet18"], help="Classifier Type")

    # Data Preparation parameters
    parser.add_argument("-dataset", type=datasets2hi, required=True, help="Dataset employed in the experiment")
    parser.add_argument("-classes", type=str, required=True, help="Comma-separated list of classes")
    parser.add_argument("-class_types", type=str, required=True, help="Comma-separated list of class types ('Base'/'Masked')")
    
    return parser.parse_args()

def validate_and_process_args(args, logger) -> dict:
    test_id = args.test_id
    
    # Check if the Metadata JSON file for the given 'test_id' already exists
    exp_metadata_path = os.path.join(LOG_ROOT, f"{test_id}-metadata.json")
    if os.path.exists(exp_metadata_path):
        logger.warning(f"*** IN RELATION TO THE EXPERIMENT '{test_id}', THE METADATA FILE HAS ALREADY BEEN CREATED! ***\n") 
        exit(1)
    
    classes = [int(c.strip()) for c in args.classes.split(',')]
    class_types = [ct.strip() for ct in args.class_types.split(',')]

    if len(classes) > len(class_types): raise Exception("There is at least one Class without the corresponding Class Type")
    if len(classes) < len(class_types): raise Exception("There is at least one Class Type which cannot be attributed to any Class")

    # Build the Metadata dictionary for the current Experiment
    exp_metadata = {"TEST_ID": test_id, "MODEL_TYPE": args.model_type, "DATASET": args.dataset, "CLASSES": dict(zip(classes, class_types))}
    return exp_metadata

### #### ###
### MAIN ###
### #### ###
if __name__ == '__main__':
    os.makedirs(LOG_ROOT, exist_ok=True)
    args = get_args()
    TEST_ID = args.test_id
    EXP_METADATA_PATH = os.path.join(LOG_ROOT, f"{TEST_ID}-metadata.json")
    logger = get_logger(TEST_ID)
    
    try:
        EXP_METADATA = validate_and_process_args(args, logger)
        save_metadata(EXP_METADATA, EXP_METADATA_PATH)
        logger.info(f"*** METADATA SUCCESSFULLY CREATED FOR: {TEST_ID} ***\n")
    except Exception as e: 
        logger.error(f"Error on Metadata creation for Experiment: {TEST_ID}")
        logger.error(f"Details: {e}")