import torch, os, pickle, json
from PIL import Image
from datetime import datetime
from argparse import ArgumentParser

from .explainers import LimeBaseExplainer, get_crops_bbxs

from .utils.model_utils import *
from .utils.image_utils import load_rgb_mean_std
from .utils.explanations_utils import get_instances_to_explain, generate_instance_mask

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
XAI_ROOT = "./xai"

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def get_args():
    parser = ArgumentParser(description="Hyperparameters for the Explanation process", add_help=True)
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-block_width", type=int, required=True)
    parser.add_argument("-block_height", type=int, required=True)
    parser.add_argument("-crop_size", type=int)
    parser.add_argument("-algorithm", type=str, required=True, choices=["LimeBase", "GLimeBinomial"])
    parser.add_argument("-surrogate_model", type=str, required=True, choices=["LinReg", "Ridge"])
    parser.add_argument("-lime_iters", type=int, default=1)
    parser.add_argument("-remove_patches", type=str, default="False")

    return parser.parse_args()

def load_metadata(metadata_path) -> dict:
    try:
        with open(metadata_path, 'r') as jf: return json.load(jf)
    except json.JSONDecodeError as e: raise ValueError(f"Error occurred while decoding JSON file '{metadata_path}': {e}")
    except Exception as e: raise FileNotFoundError(f"Error occurred while reading metadata file '{metadata_path}': {e}")

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
        
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]
    BLOCK_WIDTH, BLOCK_HEIGHT = args.block_width, args.block_height
    CROP_SIZE = args.crop_size
    XAI_ALGORITHM = args.algorithm
    SURROGATE_MODEL = args.surrogate_model
    LIME_ITERS = args.lime_iters
    REMOVE_PATCHES = args.remove_patches

    CLASSIFIER_ROOT = f"./classifiers/classifier_{MODEL_TYPE}"
    cp_base = f"{CLASSIFIER_ROOT}/../cp/Test_3_TL_val_best_model.pth"
    cp = f"{CLASSIFIER_ROOT}/tests/{TEST_ID}/output/checkpoints/Test_{TEST_ID}_MLC_val_best_model.pth"

    CLASSES_DATA = EXP_METADATA["CLASSES"]
    classes = list(CLASSES_DATA.keys())
    num_classes = len(classes)
        
    start, dir_name = None, None
    if f"{XAI_ALGORITHM}_METADATA" in EXP_METADATA:
        if "EXP_END_TIMESTAMP" in EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]:
            print("*** IN RELATION TO THE SPECIFIED EXPERIMENT, THE MODEL PREDICTIONS HAVE ALREADY BEEN EXPLAINED WITH THE SPECIFIED ALGORITHM! ***\n")
            exit(1)
        
        start = EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["EXP_START_TIMESTAMP"]
        dir_name = EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["DIR_NAME"]
        
    else:
        start = str(datetime.now())
        start = str.replace(start, '-', '.')
        start = str.replace(start, ' ', '_')
        start = str.replace(start, ':', '.')
        
        dir_name = TEST_ID + "-" + MODEL_TYPE + "-" + start + "-" + XAI_ALGORITHM + "-" + SURROGATE_MODEL
        
        EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"] = dict()
        EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["EXP_START_TIMESTAMP"] = start
        EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["BLOCK_DIM"] = {"WIDTH": BLOCK_WIDTH, "HEIGHT": BLOCK_HEIGHT}
        EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["DIR_NAME"] = dir_name
        EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["INSTANCES"] = dict()
        with open(EXP_METADATA_PATH, 'w') as jf: json.dump(EXP_METADATA, jf, indent=4)
    
    print("*** BEGINNING OF EXPLAINABILITY PROCESS ***")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda": print("Device:", torch.cuda.get_device_name(0))
    else: print("Device: CPU")
    
    model = None
    match MODEL_TYPE:
        case 'NN':
            model = NN_Classifier(num_classes=num_classes, mode='frozen', cp_path=cp_base)
            model.load_state_dict(torch.load(cp)['model_state_dict'])
            model.eval()
            model.to(DEVICE)
    print("Model Loaded!")
    print("Classes:", classes)
    
    mean, std = load_rgb_mean_std(f"{CLASSIFIER_ROOT}/tests/{TEST_ID}")
    

    explainer = None
    if XAI_ALGORITHM == "LimeBase":    
        explainer = LimeBaseExplainer(classifier=f"classifier_{MODEL_TYPE}",
                                        test_id=TEST_ID,
                                        dir_name=dir_name,
                                        block_size=(BLOCK_WIDTH, BLOCK_HEIGHT),
                                        model=model,
                                        surrogate_model=SURROGATE_MODEL,
                                        mean=mean,
                                        std=std,
                                        device=DEVICE)
    
    ### ################### ###
    ### EXPLANATION PROCESS ###
    ### ################### ###
    BASE_ID, RET_ID = None, None
    try: BASE_ID, RET_ID = TEST_ID.split(':')
    except: BASE_ID = TEST_ID
    
    DATASET = EXP_METADATA["DATASET"]
    SOURCE_DATA_DIR = f"{DATASET_ROOT}/{DATASET}/processed"
    instances, labels = list(), list()
    
    class_to_idx = None
    with open(f"{CLASSIFIER_ROOT}/tests/{TEST_ID}/output/class_to_idx.pkl", "rb") as f: class_to_idx = pickle.load(f)

    for c, c_type in CLASSES_DATA.items():
        class_source = None
        if c_type == "base": class_source = SOURCE_DATA_DIR + f"/{c}"
        else: class_source = SOURCE_DATA_DIR + f"/{c}-{BASE_ID}_{MODEL_TYPE}_{c_type}"
        
        if "ret" not in TEST_ID:
            train_instances, train_labels = get_instances_to_explain(DATASET, class_source, class_to_idx, "train")
            instances.extend(train_instances)
            labels.extend(train_labels)
        
        test_instances, test_labels = get_instances_to_explain(DATASET, class_source, class_to_idx, "test")
        instances.extend(test_instances)
        labels.extend(test_labels)
        
    # 2 -> Explanation Process
    OVERLAP = CROP_SIZE - 25

    for instance_name, label in zip(instances, labels):
        if instance_name in EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["INSTANCES"]:
            print(f"Skipping Instance '{instance_name}' with label '{label}': already explained!")
            os.system(f"rm {XAI_ROOT}/data/{instance_name}")
            continue
        
        print(f"Processing Instance '{instance_name}' with label '{label}'")
        
        img_path = f"{XAI_ROOT}/data/{instance_name}"
        mask_path = f"{XAI_ROOT}/def_mask_{BLOCK_WIDTH}x{BLOCK_HEIGHT}.png"
        
        if not os.path.exists(mask_path): generate_instance_mask(inst_width=902, inst_height=1279, block_width=BLOCK_WIDTH, block_height=BLOCK_HEIGHT)
            
        img, mask = Image.open(img_path), Image.open(mask_path)

        # 3.1 -> Feature Attribution for the Instance Superpixels (identified by the Page Mask)
        explainer.compute_superpixel_scores(img, mask, instance_name, label, LIME_ITERS, CROP_SIZE, OVERLAP)
        explainer.visualize_superpixel_scores_outcomes(img, mask, instance_name, reduction_method="mean", min_eval=2)

        # 3.2 -> Generating the Explanation for the Instance Crops by removing "Relevant", "Misleading" and "Random" Patches
        if REMOVE_PATCHES == "True":
            crops_bbxs = get_crops_bbxs(img, final_width=CROP_SIZE, final_height=CROP_SIZE)
            explainer.compute_masked_patches_explanation(img, mask, instance_name, label, crops_bbxs, reduction_method="mean", min_eval=10, num_samples_for_baseline=10)
        
        EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["INSTANCES"][f"{instance_name}"] = str(datetime.now())
        with open(EXP_METADATA_PATH, 'w') as jf: json.dump(EXP_METADATA, jf, indent=4)
        os.system(f"rm {XAI_ROOT}/data/{instance_name}")
    
    os.system(f"cp {CLASSIFIER_ROOT}/tests/{TEST_ID}/rgb_train_stats.pkl {XAI_ROOT}/explanations/patches_{BLOCK_WIDTH}x{BLOCK_HEIGHT}_removal/{dir_name}/rgb_train_stats.pkl")
    EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["EXP_END_TIMESTAMP"] = str(datetime.now())
    with open(EXP_METADATA_PATH, 'w') as jf: json.dump(EXP_METADATA, jf, indent=4)
    
    print("All the requested Instances have been explained!\n")
    print("*** END OF EXPLAINABILITY PROCESS ***\n")