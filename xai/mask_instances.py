from argparse import ArgumentParser

from utils import load_metadata

from xai.maskers.image_masker import SaliencyMasker, RandomMasker
from xai.utils.mask_utils import setup_masking_process

LOG_ROOT = "./log"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase"])
    parser.add_argument("-xai_mode", type=str, required=True, choices=["base", "counterfactual_top_class"])
    parser.add_argument("-instances", type=str, required=True, choices=["train", "test"])
    parser.add_argument("-mask_rate", type=float, required=True)
    parser.add_argument("-mask_mode", type=str, required=True, choices=["saliency", "random"])

    return parser.parse_args()
 
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
    
    XAI_ALGORITHM = args.xai_algorithm
    XAI_MODE = args.xai_mode
    INSTANCE_SET = args.instances
    MASK_RATE = args.mask_rate
    MASK_MODE = args.mask_mode
        
    if f"{XAI_ALGORITHM}_{XAI_MODE}_METADATA" not in EXP_METADATA:
        print(f"*** IN RELATION TO THE '{TEST_ID}' EXPERIMENT, THE XAI ALGORITHM '{XAI_ALGORITHM}' WITH MODE '{XAI_MODE}' HAS NOT BEEN EMPLOYED YET! ***\n")
        exit(1)
    
    if f"MASK_PROCESS_{INSTANCE_SET}_{MASK_MODE}_{MASK_RATE}_{XAI_ALGORITHM}_{XAI_MODE}_END_TIMESTAMP" in EXP_METADATA:
        print(f"*** IN RELATION TO ['{TEST_ID}' EXPERIMENT, '{XAI_ALGORITHM}-{XAI_MODE}' XAI ALGORITHM, '{MASK_MODE}' MODE, '{MASK_RATE}' MASK RATE], THE MASKING PROCESS HAS ALREADY BEEN PERFORMED! ***\n")
        exit(1)
    
    DATASET = EXP_METADATA["DATASET"]
    CLASSES = list(EXP_METADATA["CLASSES"].keys())
    EXP_DIR = EXP_METADATA[f"{XAI_ALGORITHM}_{XAI_MODE}_METADATA"]["DIR_NAME"]
    PATCH_WIDTH = EXP_METADATA[f"{XAI_ALGORITHM}_{XAI_MODE}_METADATA"]["PATCH_DIM"]["WIDTH"]
    PATCH_HEIGHT = EXP_METADATA[f"{XAI_ALGORITHM}_{XAI_MODE}_METADATA"]["PATCH_DIM"]["HEIGHT"]
    
    instances, paths = setup_masking_process(DATASET, CLASSES, INSTANCE_SET)
    
    masker = None
    if MASK_MODE == "saliency":
        masker = SaliencyMasker(dataset=DATASET,inst_set=INSTANCE_SET, instances=instances, 
            paths=paths, test_id=TEST_ID, exp_dir=EXP_DIR, mask_rate=MASK_RATE, 
            mask_mode=MASK_MODE, patch_width=PATCH_WIDTH, patch_height=PATCH_HEIGHT,
            xai_algorithm=XAI_ALGORITHM, xai_mode=XAI_MODE, exp_metadata=EXP_METADATA, save_patches=True, verbose=True)
    elif MASK_MODE == "random":
        masker = RandomMasker(dataset=DATASET,inst_set=INSTANCE_SET, instances=instances, 
            paths=paths, test_id=TEST_ID, exp_dir=EXP_DIR, mask_rate=MASK_RATE, 
            mask_mode=MASK_MODE, patch_width=PATCH_WIDTH, patch_height=PATCH_HEIGHT,
            xai_algorithm=XAI_ALGORITHM, xai_mode=XAI_MODE, exp_metadata=EXP_METADATA, save_patches=False, verbose=True)
    
    masker()