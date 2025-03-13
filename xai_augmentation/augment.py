import os, torch
from argparse import ArgumentParser
from datetime import datetime

from utils import xaiaug2str, str2bool, load_metadata, save_metadata

from classifiers.utils.fine_tune_utils import load_model
from classifiers.utils.dataloader_utils import Faithfulness_Test_DataLoader, load_rgb_mean_std

from xai_augmentation.augment_utils import retrieve_training_instances, setup_dimensionality_reduction, extract_class_mean_vectors, compute_augmentation_rates, extract_augmented_crops

LOG_ROOT = "./log"
CLASSIFIERS_ROOT = "./classifiers"
XAI_ROOT = "./xai"
XAI_AUG_ROOT = "./xai_augmentation"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase"])
    parser.add_argument("-mode", type=xaiaug2str, required=True)
    parser.add_argument("-keep_balanced", type=str2bool, default=True)
    
    return parser.parse_args()
                  
### #### ###
### MAIN ###
### #### ###
if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = get_args()
    
    TEST_ID = args.test_id
    XAI_ALGORITHM = args.xai_algorithm
    MODE = args.mode
    KEEP_BALANCED = args.keep_balanced

    EXP_METADATA = load_metadata(f"{LOG_ROOT}/{TEST_ID}-metadata.json")
    DATASET = EXP_METADATA["DATASET"]
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]
    CLASSES_DATA = EXP_METADATA["CLASSES"]
    CROP_SIZE = EXP_METADATA["FINE_TUNING_HP"]["crop_size"]
    
    CLASSES = list(CLASSES_DATA.keys())
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CP = f"{CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/cp/Test_3_TL_val_best_model.pth"
    
    model, _ = load_model(MODEL_TYPE, len(CLASSES_DATA), "frozen", CP, "test", TEST_ID, EXP_METADATA, DEVICE)

    root_dir = f"{XAI_AUG_ROOT}/{TEST_ID}/{MODE}_balanced" if KEEP_BALANCED else f"{XAI_AUG_ROOT}/{TEST_ID}/{MODE}_notbalanced"
        
    os.makedirs(root_dir, exist_ok=True)
    retrieve_training_instances(root_dir, DATASET, CLASSES)
    EXP_DIR = f"{CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/tests/{TEST_ID}"
    mean_, std_ = load_rgb_mean_std(EXP_DIR)
    dl = Faithfulness_Test_DataLoader(directory=f"{root_dir}/train_instances", classes=CLASSES, batch_size=1, img_crop_size=CROP_SIZE, mean=mean_, std=std_)

    print(f"*** BEGINNING OF AUGMENTATION PROCESS: MODE '{MODE}', KEEP_BALANCED '{KEEP_BALANCED}' ***")
    
    pca = None
    if MODE in ["pi", "wo"]:
        pca = setup_dimensionality_reduction(model, dl, DEVICE, final_dim=64)
    
    print("Extracting class mean vectors...")
    mean_class_vectors, c_to_idx, idx_to_c = extract_class_mean_vectors(model, dl, pca, DEVICE)
    
    print("Computing augmentation rates...")
    augmentations = compute_augmentation_rates(model, dl, idx_to_c, DEVICE)
    
    os.makedirs(f"{root_dir}/crops_for_augmentation", exist_ok=True)
    
    XAI_EXP_DIR_NAME = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["DIR_NAME"]
    P_WIDTH, P_HEIGHT = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["PATCH_DIM"]["WIDTH"], EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["PATCH_DIM"]["HEIGHT"]
    XAI_EXP_DIR = f"{XAI_ROOT}/explanations/patches_{P_WIDTH}x{P_HEIGHT}_removal/{XAI_EXP_DIR_NAME}"

    if KEEP_BALANCED:
        max_augmentations = 0
        for c in CLASSES:
            n_augmentations = augmentations[c_to_idx[str(c)]]
            if n_augmentations > max_augmentations: max_augmentations = n_augmentations
        
        if MODE == "rand":
            for c in CLASSES: augmentations[c_to_idx[str(c)]] = (0, max_augmentations)
        else:
            for c in CLASSES:
                xai_augmentations = augmentations[c_to_idx[str(c)]]
                random_augmentations = max_augmentations - xai_augmentations
                augmentations[c_to_idx[str(c)]] = (xai_augmentations, random_augmentations)
    
    else:
        if MODE == "rand":
            for c in CLASSES:
                class_augmentations = augmentations[c_to_idx[str(c)]]
                augmentations[c_to_idx[str(c)]] = (0, class_augmentations)
        else:
            for c in CLASSES:
                xai_augmentations = augmentations[c_to_idx[str(c)]]
                augmentations[c_to_idx[str(c)]] = (xai_augmentations, 0)

    for c in CLASSES:
        xai_and_random_augmentations = augmentations[c_to_idx[str(c)]]
        extract_augmented_crops(MODE, root_dir, XAI_EXP_DIR, DATASET, c, c_to_idx, mean_class_vectors, xai_and_random_augmentations, CROP_SIZE, P_WIDTH, P_HEIGHT, mean_, std_, pca, model, DEVICE)
        
    xai_aug_metadata_entry = f"XAI_AUGMENTATION_{MODE}_BALANCED" if KEEP_BALANCED else f"XAI_AUGMENTATION_{MODE}"
    EXP_METADATA[xai_aug_metadata_entry] = dict()
    EXP_METADATA[xai_aug_metadata_entry]["TIMESTAMP"] = str(datetime.now())
    EXP_METADATA[xai_aug_metadata_entry]["INSTANCES"] = dict()
    for c in CLASSES:
        EXP_METADATA[xai_aug_metadata_entry]["INSTANCES"][f"class_{c}"] = dict()
        EXP_METADATA[xai_aug_metadata_entry]["INSTANCES"][f"class_{c}"]["XAI_AUG_CROPS"] = augmentations[c_to_idx[str(c)]][0]
        EXP_METADATA[xai_aug_metadata_entry]["INSTANCES"][f"class_{c}"]["RANDOM_AUG_CROPS"] = augmentations[c_to_idx[str(c)]][1]
    save_metadata(EXP_METADATA, f"{LOG_ROOT}/{TEST_ID}-metadata.json")
    os.system(f"rm -r {root_dir}/train_instances")
    
    print(f"*** END OF AUGMENTATION PROCESS WITH: MODE '{MODE}', KEEP_BALANCED '{KEEP_BALANCED}' ***\n")