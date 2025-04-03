import torch, os, json, pickle
from datetime import datetime
from argparse import ArgumentParser
from PIL import Image

from utils import load_metadata, save_metadata, get_model_base_checkpoint
from classifiers.utils.fine_tune_utils import load_model

from xai.utils.image_utils import produce_padded_page, generate_mask
from xai.utils.explanations_utils import get_instances_to_explain, setup_explainer, explain_instance, visualize_exp_outcome

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
CLASSIFIERS_ROOT = "./classifiers"
XAI_ROOT = "./xai"

def get_args():
    parser = ArgumentParser(description="Hyperparameters for the Explanation process", add_help=True)
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-patch_width", type=int, required=True)
    parser.add_argument("-patch_height", type=int, required=True)
    parser.add_argument("-overlap", type=float, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase", "GLimeBinomial"])
    parser.add_argument("-surrogate_model", type=str, required=True, choices=["LinReg", "Ridge", "Lasso", "ElasticNet"])
    parser.add_argument("-num_samples", type=int, default=100)
    parser.add_argument("-kernel_width", type=float, default=None)
    parser.add_argument("-mode", type=str, default="base", choices=["base", "counterfactual_top_class"])
    parser.add_argument("-iters", type=int, default=1)

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
    OVERLAP = int(args.overlap * CROP_SIZE)
    XAI_ALGORITHM = args.xai_algorithm
    SURROGATE_MODEL = args.surrogate_model
    NUM_SAMPLES = args.num_samples
    KERNEL_WIDTH = args.kernel_width
    if KERNEL_WIDTH is None:
        if XAI_ALGORITHM == "LimeBase": KERNEL_WIDTH = 5.0
        elif XAI_ALGORITHM == "GLimeBinomial": KERNEL_WIDTH = 0.5
    MODE = args.mode
    ITERS = args.iters

    DATASET = EXP_METADATA["DATASET"]
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]
    CLASSES_DATA = EXP_METADATA["CLASSES"]
    classes = list(CLASSES_DATA.keys())
    num_classes = len(classes)
    
    cp_base = get_model_base_checkpoint(MODEL_TYPE)
        
    start, dir_name = None, None
    if f"{XAI_ALGORITHM}_{MODE}_{SURROGATE_MODEL}_METADATA" in EXP_METADATA:
        if "XAI_END_TIMESTAMP" in EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_{SURROGATE_MODEL}_METADATA"]:
            print(f"*** IN RELATION TO THE EXPERIMENT '{TEST_ID}', THE MODEL PREDICTIONS HAVE ALREADY BEEN EXPLAINED WITH '{XAI_ALGORITHM}', WITH MODE '{MODE}' AND USING '{SURROGATE_MODEL}' AS SURROGATE MODEL! ***\n")
            exit(1)
        
        start = EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_{SURROGATE_MODEL}_METADATA"]["XAI_START_TIMESTAMP"]
        dir_name = EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_{SURROGATE_MODEL}_METADATA"]["DIR_NAME"]
        
        XAI_METADATA_PATH = f"{XAI_ROOT}/explanations/patches_{PATCH_WIDTH}x{PATCH_HEIGHT}_removal/{dir_name}/{TEST_ID}-xai_metadata.json"
        XAI_METADATA = load_metadata(XAI_METADATA_PATH)
        
    else:
        start = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        dir_name = f"{TEST_ID}-{MODEL_TYPE}-{start}-{XAI_ALGORITHM}-{SURROGATE_MODEL}"
        
        explanations_dir_path = f"{XAI_ROOT}/explanations/patches_{PATCH_WIDTH}x{PATCH_HEIGHT}_removal/{dir_name}"
        os.makedirs(explanations_dir_path, exist_ok=True)
        
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_{SURROGATE_MODEL}_METADATA"] = dict()
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_{SURROGATE_MODEL}_METADATA"]["XAI_START_TIMESTAMP"] = start
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_{SURROGATE_MODEL}_METADATA"]["PATCH_DIM"] = {"WIDTH": PATCH_WIDTH, "HEIGHT": PATCH_HEIGHT}
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_{SURROGATE_MODEL}_METADATA"]["NUM_SAMPLES"] = NUM_SAMPLES
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_{SURROGATE_MODEL}_METADATA"]["OVERLAP"] = OVERLAP
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_{SURROGATE_MODEL}_METADATA"]["ITERS"] = ITERS
        EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_{SURROGATE_MODEL}_METADATA"]["DIR_NAME"] = dir_name
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
    
    explainer = setup_explainer(XAI_ALGORITHM, SURROGATE_MODEL, MODEL_TYPE, model, NUM_SAMPLES, KERNEL_WIDTH, mean, std)
    
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
    os.makedirs(f"{XAI_ROOT}/masks", exist_ok=True)
    os.makedirs(f"{XAI_ROOT}/explanations/patches_{PATCH_WIDTH}x{PATCH_HEIGHT}_removal/{dir_name}", exist_ok=True)
    
    for instance_path, label in zip(instances, labels):
        instance_name = instance_path.split("/")[-1][:-4]
        instance_type = instance_path.split("/")[-1][-4:]
        
        if instance_name in XAI_METADATA["INSTANCES"]:
            print(f"Skipping instance '{instance_name}' with label '{label}': already explained!")
        
        else:
            print(f"Processing Instance '{instance_name}' with label '{label}'")
            output_dir = f"{XAI_ROOT}/explanations/patches_{PATCH_WIDTH}x{PATCH_HEIGHT}_removal/{dir_name}/{instance_name}"
            os.makedirs(output_dir, exist_ok=True)
            img = Image.open(instance_path)
            padded_img, img_rows, img_cols = produce_padded_page(img, instance_name, instance_type, CROP_SIZE, OVERLAP, mean, output_dir)

            mask_dir = f"{XAI_ROOT}/masks/{DATASET}_mask_{PATCH_WIDTH}x{PATCH_HEIGHT}_cs{CROP_SIZE}_overlap{OVERLAP}"
            os.makedirs(mask_dir, exist_ok=True)
            if not os.path.exists(f"{mask_dir}/mask.png"):
                mask_img, mask_rows, mask_cols = generate_mask(padded_img, DATASET, PATCH_WIDTH, PATCH_HEIGHT, CROP_SIZE, OVERLAP)
            else: 
                mask_img = Image.open(f"{mask_dir}/mask.png")
                with open(f"{mask_dir}/dimensions.json", "r") as f: dimensions = json.load(f)
                mask_rows, mask_cols = dimensions["mask_rows"], dimensions["mask_cols"]

            norm_scores = None
            if f"{instance_name}_scores.pkl" not in os.listdir(f"{output_dir}"):
                norm_scores = explain_instance(explainer, padded_img, img_rows, img_cols, instance_name, label, mask_img, mask_rows, mask_cols, output_dir, CROP_SIZE, OVERLAP, ITERS)
            else: 
                with open(f"{output_dir}/{instance_name}_scores.json", "r") as f: norm_scores = json.load(f)
            
            visualize_exp_outcome(norm_scores, mask_img, instance_name, output_dir)
            
            XAI_METADATA["INSTANCES"][f"{instance_name}"] = dict()
            XAI_METADATA["INSTANCES"][f"{instance_name}"]["label"] = label
            XAI_METADATA["INSTANCES"][f"{instance_name}"]["timestamp"] = str(datetime.now())
            save_metadata(XAI_METADATA, XAI_METADATA_PATH)
    
    os.system(f"cp {CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/tests/{TEST_ID}/rgb_train_stats.pkl {XAI_ROOT}/explanations/patches_{PATCH_WIDTH}x{PATCH_HEIGHT}_removal/{dir_name}/rgb_train_stats.pkl")
    EXP_METADATA[f"{XAI_ALGORITHM}_{MODE}_METADATA"]["XAI_END_TIMESTAMP"] = str(datetime.now())
    save_metadata(EXP_METADATA, EXP_METADATA_PATH)
    
    print("All the requested Instances have been explained!\n")
    print("*** END OF EXPLAINABILITY PROCESS ***\n")