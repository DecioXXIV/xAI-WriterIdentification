import os, json
from argparse import ArgumentParser

LOG_ROOT = "./log"

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def get_args():
    parser = ArgumentParser()
    
    # General parameters
    parser.add_argument("-test_id", type=str, required=True, help="ID for the new Experiment to be performed")
    parser.add_argument("-model_type", type=str, required=True, choices=['NN', 'GB', 'SVM'], help="Classifier Type")

    # Data Preparation parameters
    parser.add_argument("-dataset", type=str, required=True, choices=['CEDAR_Letter', 'CVL'], help="Dataset employed in the experiment")
    parser.add_argument("-classes", type=str, required=True, help="Comma-separated list of classes")
    parser.add_argument("-class_types", type=str, required=True, help="Comma-separated list of class types ('Base'/'Masked')")
    parser.add_argument("-vert_mult_fact", type=int, default=1, help="Multiplicative factor for vertical cuts")
    parser.add_argument("-hor_mult_fact", type=int, default=1, help="Multiplicative factor for horizontal cuts")
    
    return parser.parse_args()

def validate_and_process_args(args) -> dict:
    test_id = args.test_id
    
    # Check if the Metadata JSON file for the given 'test_id' already exists
    if os.path.exists(f"{LOG_ROOT}/{test_id}-metadata.json"): 
        print("*** IN RELATION TO THE SPECIFIED EXPERIMENT, THE METADATA FILE HAS ALREADY BEEN CREATED! ***\n")
        exit(1)
    
    # Parse and validation for the 'classes' and the 'class_types' parameters
    args_classes, classes = args.classes.split(','), list()
    for a in args_classes: classes.append(int(a.strip()))
    
    args_class_types, class_types = args.class_types.split(','), list()
    for a in args_class_types: class_types.append(a.strip())

    if len(classes) > len(class_types): raise Exception("There is at least one Class without the corresponding Class Type")
    if len(classes) < len(class_types): raise Exception("There is at least one Class Type which cannot be attributed to any Class")

    # Build the Metadata dictionary for the current Experiment
    exp_metadata = {
        "TEST_ID": test_id,
        "MODEL_TYPE": args.model_type,
        "DATASET": args.dataset,
        "CLASSES": {c: c_type for c, c_type in zip(classes, class_types)},
        "PREP_MULT_FACT": {
            "VERT": args.vert_mult_fact,
            "HOR": args.hor_mult_fact
        } 
    }

    return exp_metadata

def save_metadata(metadata: dict, test_id: str):
    with open(f"{LOG_ROOT}/{test_id}-metadata.json", "w") as jf: json.dump(metadata, jf, indent=4)

### #### ###
### MAIN ###
### #### ###
if __name__ == '__main__':
    args = get_args()
    TEST_ID = args.test_id
    
    try:
        exp_metadata = validate_and_process_args(args)
        save_metadata(exp_metadata, TEST_ID)
        print(f"*** METADATA SUCCESSFULLY CREATED FOR: {TEST_ID} ***\n")
    except Exception as e: print(f"Error on Metadata creation for Experiment: {TEST_ID}")