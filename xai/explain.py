import torch, os, pickle
from PIL import Image
from datetime import datetime
from argparse import ArgumentParser

from utils import load_metadata, save_metadata
from classifiers.classifier_NN.model import NN_Classifier

from .explainers.lime_base_explainer import LimeBaseExplainer
from .utils.image_utils import generate_instance_mask, load_rgb_mean_std, get_crops_bbxs
from .utils.explanations_utils import get_instances_to_explain

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
    parser.add_argument("-crop_size", type=int, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase", "GLimeBinomial"])
    parser.add_argument("-surrogate_model", type=str, required=True, choices=["LinReg", "Ridge"])
    parser.add_argument("-mode", type=str, required=True, choices=["base", "counterfactual_top_class"])
    parser.add_argument("-lime_iters", type=int, default=3)
    parser.add_argument("-remove_patches", type=str, default="False")

    return parser.parse_args()

def load_model(model_type, num_classes, cp_base, checkpoint_path, device):
    if model_type == "NN":
        model = NN_Classifier(num_classes=num_classes, mode="frozen", cp_path=cp_base)
        model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
        model.eval()
        model.to(device)
        
        return model

def setup_explainer(xai_algorithm, args, model_type, model, dir_name, mean, std, device):
    explainer = None
    if xai_algorithm == "LimeBase":
        explainer = LimeBaseExplainer(
            classifier=f"classifier_{model_type}",
            test_id=args.test_id,
            dir_name=dir_name,
            block_size=(args.block_width, args.block_height),
            model=model,
            surrogate_model=args.surrogate_model,
            mean=mean,
            std=std,
            device=device
        )
    
    return explainer

def explain_instance(instance_name, label, explainer, crop_size, overlap, lime_iters, xai_metadata, remove_patches):
    if instance_name in xai_metadata["INSTANCES"]:
        print(f"Skipping Instance '{instance_name}' with label '{label}': already explained!")
        os.system(f"rm {XAI_ROOT}/data/{instance_name}")
        return
    
    print(f"Processing Instance '{instance_name}' with label '{label}'")
    
    img_path = f"{XAI_ROOT}/data/{instance_name}"
    mask_path = f"{XAI_ROOT}/def_mask_{BLOCK_WIDTH}x{BLOCK_HEIGHT}.png"
    if not os.path.exists(mask_path): generate_instance_mask(inst_width=902, inst_height=1279, block_width=BLOCK_WIDTH, block_height=BLOCK_HEIGHT)
    img, mask = Image.open(img_path), Image.open(mask_path)
    
    explainer.compute_superpixel_scores(img, mask, instance_name, label, lime_iters, crop_size, overlap)
    explainer.visualize_superpixel_scores_outcomes(img, mask, instance_name, reduction_method="mean", min_eval=2)
    
    ### To Fix! ###
    if remove_patches == "True":
        crops_bbxs = get_crops_bbxs(img, crop_size, crop_size)
        explainer.compute_masked_patches_explanation(img, mask, instance_name, label, crops_bbxs, reduction_method="mean", min_eval=10, num_samples_for_baseline=10)
    ### ####### ###
    
    xai_metadata["INSTANCES"][f"{instance_name}"] = dict()
    xai_metadata["INSTANCES"][f"{instance_name}"]["label"] = label
    xai_metadata["INSTANCES"][f"{instance_name}"]["timestamp"] = str(datetime.now())
    save_metadata(xai_metadata, XAI_METADATA_PATH)
    os.system(f"rm {XAI_ROOT}/data/{instance_name}")
    

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
    XAI_ALGORITHM = args.xai_algorithm
    SURROGATE_MODEL = args.surrogate_model
    MODE = args.mode
    LIME_ITERS = args.lime_iters
    REMOVE_PATCHES = args.remove_patches

    CLASSIFIER_ROOT = f"./classifiers/classifier_{MODEL_TYPE}"
    cp_base = f"{CLASSIFIER_ROOT}/../cp/Test_3_TL_val_best_model.pth"
    cp = f"{CLASSIFIER_ROOT}/tests/{TEST_ID}/output/checkpoints/Test_{TEST_ID}_MLC_val_best_model.pth"

    CLASSES_DATA = EXP_METADATA["CLASSES"]
    classes = list(CLASSES_DATA.keys())
    num_classes = len(classes)
        
    start, dir_name = None, None
    if f"{XAI_ALGORITHM}_{MODE}_METADATA" in EXP_METADATA:
        if "XAI_END_TIMESTAMP" in EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]:
            print(f"*** IN RELATION TO THE EXPERIMENT '{TEST_ID}', THE MODEL PREDICTIONS HAVE ALREADY BEEN EXPLAINED WITH '{XAI_ALGORITHM}' AND UNDER MODE '{MODE}'! ***\n")
            exit(1)
        
        start = EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["XAI_START_TIMESTAMP"]
        dir_name = EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["DIR_NAME"]
        
        XAI_METADATA_PATH = f"{XAI_ROOT}/explanations/patches_{BLOCK_WIDTH}x{BLOCK_HEIGHT}_removal/{dir_name}/{TEST_ID}-xai_metadata.json"
        XAI_METADATA = load_metadata(XAI_METADATA_PATH)
        
    else:
        start = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        dir_name = f"{TEST_ID}-{MODEL_TYPE}-{start}-{XAI_ALGORITHM}-{SURROGATE_MODEL}"
        
        explanations_dir_path = f"{XAI_ROOT}/explanations/patches_{BLOCK_WIDTH}x{BLOCK_HEIGHT}_removal/{dir_name}"
        os.makedirs(explanations_dir_path, exist_ok=True)
        
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"] = dict()
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["XAI_START_TIMESTAMP"] = start
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["BLOCK_DIM"] = {"WIDTH": BLOCK_WIDTH, "HEIGHT": BLOCK_HEIGHT}
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["DIR_NAME"] = dir_name
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["REMOVE_PATCHES"] = REMOVE_PATCHES
        save_metadata(EXP_METADATA, EXP_METADATA_PATH)
        
        XAI_METADATA = dict()
        XAI_METADATA["INSTANCES"] = dict()
        XAI_METADATA_PATH = f"{XAI_ROOT}/explanations/patches_{BLOCK_WIDTH}x{BLOCK_HEIGHT}_removal/{dir_name}/{TEST_ID}-xai_metadata.json"
        save_metadata(XAI_METADATA, XAI_METADATA_PATH) 
    
    print("*** BEGINNING OF EXPLAINABILITY PROCESS ***")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda": print("Device:", torch.cuda.get_device_name(0))
    else: print("Device: CPU")
    
    model = load_model(MODEL_TYPE, num_classes, cp_base, cp, DEVICE)
    print("Model Loaded!")
    print("Classes:", classes)
    
    mean, std = load_rgb_mean_std(f"{CLASSIFIER_ROOT}/tests/{TEST_ID}")
    
    explainer = setup_explainer(XAI_ALGORITHM, args, MODEL_TYPE, model, dir_name, mean, std, DEVICE)
    
    ### Explainability Process
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
        if c_type == "base": class_source = f"{SOURCE_DATA_DIR}/{c}"
        else: class_source = f"{SOURCE_DATA_DIR}/{c}-{BASE_ID}_{MODEL_TYPE}_{c_type}"
        
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
        explain_instance(instance_name, label, explainer, CROP_SIZE, OVERLAP, LIME_ITERS, XAI_METADATA, REMOVE_PATCHES)
    
    os.system(f"cp {CLASSIFIER_ROOT}/tests/{TEST_ID}/rgb_train_stats.pkl {XAI_ROOT}/explanations/patches_{BLOCK_WIDTH}x{BLOCK_HEIGHT}_removal/{dir_name}/rgb_train_stats.pkl")
    EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["XAI_END_TIMESTAMP"] = str(datetime.now())
    save_metadata(EXP_METADATA, EXP_METADATA_PATH)
    
    print("All the requested Instances have been explained!\n")
    print("*** END OF EXPLAINABILITY PROCESS ***\n")