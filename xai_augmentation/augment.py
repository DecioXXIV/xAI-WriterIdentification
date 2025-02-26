import os, torch
from argparse import ArgumentParser

from utils import load_metadata

from classifiers.utils.fine_tune_utils import load_model
from classifiers.utils.dataloader_utils import Faithfulness_Test_DataLoader, load_rgb_mean_std

from xai_augmentation.augment_utils import retrieve_training_instances, extract_class_mean_vectors, compute_augmentation_rates, extract_augmented_crops

LOG_ROOT = "./log"
CLASSIFIERS_ROOT = "./classifiers"
XAI_ROOT = "./xai"
XAI_AUG_ROOT = "./xai_augmentation"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase"])
    
    return parser.parse_args()
                  
### #### ###
### MAIN ###
### #### ###
if __name__ == "__main__":
    args = get_args()
    
    TEST_ID = args.test_id
    XAI_ALGORITHM = args.xai_algorithm
    EXP_METADATA = load_metadata(f"{LOG_ROOT}/{TEST_ID}-metadata.json")
    DATASET = EXP_METADATA["DATASET"]
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]
    CLASSES_DATA = EXP_METADATA["CLASSES"]
    CROP_SIZE = EXP_METADATA["FINE_TUNING_HP"]["crop_size"]
    
    CLASSES = list(CLASSES_DATA.keys())
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CP = f"{CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/cp/Test_3_TL_val_best_model.pth"
    
    model, _ = load_model(MODEL_TYPE, len(CLASSES_DATA), CP, TEST_ID, EXP_METADATA, DEVICE)
        
    root_dir = f"{XAI_AUG_ROOT}/{TEST_ID}"
    os.makedirs(root_dir, exist_ok=True)
    retrieve_training_instances(root_dir, DATASET, CLASSES)
    EXP_DIR = f"{CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/tests/{TEST_ID}"
    mean_, std_ = load_rgb_mean_std(EXP_DIR)
    dl = Faithfulness_Test_DataLoader(directory=f"{root_dir}/train_instances", classes=CLASSES, batch_size=1, img_crop_size=CROP_SIZE, mean=mean_, std=std_)
    
    print("Extracting class mean vectors...")
    mean_class_vectors, c_to_idx, idx_to_c = extract_class_mean_vectors(model, dl, DEVICE)
    
    print("Computing augmentation rates...")
    augmentations = compute_augmentation_rates(model, dl, idx_to_c, DEVICE)
    
    os.makedirs(f"{root_dir}/crops_for_augmentation", exist_ok=True)
    
    XAI_EXP_DIR_NAME = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["DIR_NAME"]
    P_WIDTH, P_HEIGHT = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["PATCH_DIM"]["WIDTH"], EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["PATCH_DIM"]["HEIGHT"]
    XAI_EXP_DIR = f"{XAI_ROOT}/explanations/patches_{P_WIDTH}x{P_HEIGHT}_removal/{XAI_EXP_DIR_NAME}"
    for c in CLASSES:
        print(f"Extracting Augmentation crops for class: {c}")
        mean_class_vector = mean_class_vectors[c_to_idx[str(c)]]
        n_augmentations = augmentations[c_to_idx[str(c)]]
        extract_augmented_crops(root_dir, XAI_EXP_DIR, c, c_to_idx, mean_class_vector, n_augmentations, CROP_SIZE, P_WIDTH, P_HEIGHT, mean_, std_, model, DEVICE)