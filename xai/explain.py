import torch, os, pickle
from datetime import datetime
from argparse import ArgumentParser

from utils import load_metadata, save_metadata, str2bool, get_model_base_checkpoint
from classifiers.utils.fine_tune_utils import load_model

from xai.utils.explanations_utils import get_instances_to_explain, setup_explainer, explain_instance

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
CLASSIFIERS_ROOT = "./classifiers"
XAI_ROOT = "./xai"

def get_args():
    parser = ArgumentParser(description="Hyperparameters for the Explanation process", add_help=True)
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-patch_width", type=int, required=True)
    parser.add_argument("-patch_height", type=int, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase"])
    parser.add_argument("-surrogate_model", type=str, required=True, choices=["LinReg", "Ridge"])
    parser.add_argument("-mode", type=str, required=True, choices=["base", "counterfactual_top_class"])
    parser.add_argument("-lime_iters", type=int, default=3)
    parser.add_argument("-remove_patches", type=str2bool, default=False)

    return parser.parse_args()

### #### ###
### MAIN ###
### #### ###
if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = get_args()
    TEST_ID = args.test_id
    EXP_METADATA_PATH = f"{LOG_ROOT}/{TEST_ID}-metadata.json"
    
    try: EXP_METADATA = load_metadata(EXP_METADATA_PATH)
    except Exception as e:
        print(e)
        exit(1)
        
    PATCH_WIDTH, PATCH_HEIGHT = args.patch_width, args.patch_height
    CROP_SIZE = EXP_METADATA["FINE_TUNING_HP"]["crop_size"]
    XAI_ALGORITHM = args.xai_algorithm
    SURROGATE_MODEL = args.surrogate_model
    MODE = args.mode
    LIME_ITERS = args.lime_iters
    REMOVE_PATCHES = args.remove_patches

    DATASET = EXP_METADATA["DATASET"]
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]
    CLASSES_DATA = EXP_METADATA["CLASSES"]
    classes = list(CLASSES_DATA.keys())
    num_classes = len(classes)
    
    cp_base = get_model_base_checkpoint(MODEL_TYPE)
        
    start, dir_name = None, None
    if f"{XAI_ALGORITHM}_{MODE}_METADATA" in EXP_METADATA:
        if "XAI_END_TIMESTAMP" in EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]:
            print(f"*** IN RELATION TO THE EXPERIMENT '{TEST_ID}', THE MODEL PREDICTIONS HAVE ALREADY BEEN EXPLAINED WITH '{XAI_ALGORITHM}' AND UNDER MODE '{MODE}'! ***\n")
            exit(1)
        
        start = EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["XAI_START_TIMESTAMP"]
        dir_name = EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["DIR_NAME"]
        
        XAI_METADATA_PATH = f"{XAI_ROOT}/explanations/patches_{PATCH_WIDTH}x{PATCH_HEIGHT}_removal/{dir_name}/{TEST_ID}-xai_metadata.json"
        XAI_METADATA = load_metadata(XAI_METADATA_PATH)
        
    else:
        start = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        dir_name = f"{TEST_ID}-{MODEL_TYPE}-{start}-{XAI_ALGORITHM}-{SURROGATE_MODEL}"
        
        explanations_dir_path = f"{XAI_ROOT}/explanations/patches_{PATCH_WIDTH}x{PATCH_HEIGHT}_removal/{dir_name}"
        os.makedirs(explanations_dir_path, exist_ok=True)
        
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"] = dict()
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["XAI_START_TIMESTAMP"] = start
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["PATCH_DIM"] = {"WIDTH": PATCH_WIDTH, "HEIGHT": PATCH_HEIGHT}
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["DIR_NAME"] = dir_name
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["REMOVE_PATCHES"] = REMOVE_PATCHES
        save_metadata(EXP_METADATA, EXP_METADATA_PATH)
        
        XAI_METADATA = dict()
        XAI_METADATA["INSTANCES"] = dict()
        XAI_METADATA_PATH = f"{XAI_ROOT}/explanations/patches_{PATCH_WIDTH}x{PATCH_HEIGHT}_removal/{dir_name}/{TEST_ID}-xai_metadata.json"
        save_metadata(XAI_METADATA, XAI_METADATA_PATH) 
    
    print("*** BEGINNING OF EXPLAINABILITY PROCESS ***")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda": print("Device:", torch.cuda.get_device_name(0))
    else: print("Device: CPU")
    
    model, _ = load_model(MODEL_TYPE, num_classes, "frozen", cp_base, "test", TEST_ID, EXP_METADATA, DEVICE)
    print("Model Loaded!")
    print("Classes:", classes)
    
    mean, std = EXP_METADATA["FINE_TUNING_HP"]["mean"], EXP_METADATA["FINE_TUNING_HP"]["std"]
    
    explainer = setup_explainer(XAI_ALGORITHM, args, MODEL_TYPE, model, dir_name, mean, std, DEVICE)
    
    BASE_ID, RET_ID = None, None
    try: BASE_ID, RET_ID = TEST_ID.split(':')
    except: BASE_ID = TEST_ID
    
    DATASET = EXP_METADATA["DATASET"]
    SOURCE_DATA_DIR = f"{DATASET_ROOT}/{DATASET}/processed"

    # Retrieve the Instances (with the corresponding Labels) to be explained
    instances, labels = list(), list()
    
    class_to_idx = None
    with open(f"{CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/tests/{TEST_ID}/output/class_to_idx.pkl", "rb") as f: class_to_idx = pickle.load(f)

    for c, c_type in CLASSES_DATA.items():
        class_source = None
        if c_type == "base": class_source = f"{SOURCE_DATA_DIR}/{c}"
        else: class_source = f"{SOURCE_DATA_DIR}/{c}-{BASE_ID}_{MODEL_TYPE}_{c_type}"
        
        if "ret" not in TEST_ID:
            train_instances, train_labels = get_instances_to_explain(DATASET, class_source, class_to_idx, "train")
            instances.extend(train_instances)
            labels.extend(train_labels)
        
        test_instances, test_labels = get_instances_to_explain(DATASET, class_source, class_to_idx, "test")
        instances.extend(test_instances)
        labels.extend(test_labels)
        
    # Explanation Process
    OVERLAP = CROP_SIZE - 25
    
    for instance_path, label in zip(instances, labels):
        explain_instance(DATASET, instance_path, label, explainer, CROP_SIZE, PATCH_WIDTH, PATCH_HEIGHT, OVERLAP, LIME_ITERS, XAI_METADATA, XAI_METADATA_PATH, REMOVE_PATCHES)
    
    os.system(f"cp {CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/tests/{TEST_ID}/rgb_train_stats.pkl {XAI_ROOT}/explanations/patches_{PATCH_WIDTH}x{PATCH_HEIGHT}_removal/{dir_name}/rgb_train_stats.pkl")
    EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["XAI_END_TIMESTAMP"] = str(datetime.now())
    save_metadata(EXP_METADATA, EXP_METADATA_PATH)
    
    print("All the requested Instances have been explained!\n")
    print("*** END OF EXPLAINABILITY PROCESS ***\n")