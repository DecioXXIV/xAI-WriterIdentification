from argparse import ArgumentParser

from utils import load_metadata, get_logger

from xai import EXPLAINERS, SURROGATES, MASK_RULES, MASK_MODES
from xai.maskers.area_image_masker import AreaSaliencyMasker, AreaRandomMasker
from xai.maskers.patch_number_image_masker import PatchNumberSaliencyMasker, PatchNumberRandomMasker
from xai.utils.mask_utils import setup_masking_process

LOG_ROOT = "./log"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=EXPLAINERS)
    parser.add_argument("-xai_mode", type=str, default="base")
    parser.add_argument("-surrogate_model", type=str, required=True, choices=SURROGATES)
    parser.add_argument("-instances", type=str, required=True, choices=["train", "test"])
    parser.add_argument("-mask_rate", type=float, required=True)
    parser.add_argument("-mask_rule", type=str, required=True, choices=MASK_RULES)
    parser.add_argument("-mask_mode", type=str, required=True, choices=MASK_MODES)

    return parser.parse_args()
 
### #### ###
### MAIN ###
### #### ###
if __name__ == '__main__':
    args = get_args()
    TEST_ID = args.test_id
    EXP_METADATA_PATH = f"{LOG_ROOT}/{TEST_ID}-metadata.json"
    logger = get_logger(TEST_ID)

    try: EXP_METADATA = load_metadata(EXP_METADATA_PATH, logger)
    except Exception as e:
        logger.error(f"Failed to load Metadata for experiment {TEST_ID}")
        logger.error(f"Details: {e}")
        exit(1)
    
    XAI_ALGORITHM = args.xai_algorithm
    XAI_MODE = args.xai_mode
    SURROGATE_MODEL = args.surrogate_model
    INSTANCE_SET = args.instances
    MASK_RATE = args.mask_rate
    MASK_RULE = args.mask_rule
    MASK_MODE = args.mask_mode
        
    if f"{XAI_ALGORITHM}_{XAI_MODE}_{SURROGATE_MODEL}_METADATA" not in EXP_METADATA:
        logger.warning(f"*** IN RELATION TO THE '{TEST_ID}' EXPERIMENT, THE XAI ALGORITHM '{XAI_ALGORITHM}' WITH MODE '{XAI_MODE}' HAS NOT BEEN EMPLOYED YET! ***\n")
        exit(1)
    
    if f"MASK_PROCESS_{INSTANCE_SET}_{MASK_MODE}_{MASK_RULE}_{MASK_RATE}_{XAI_ALGORITHM}_{XAI_MODE}_{SURROGATE_MODEL}_END_TIMESTAMP" in EXP_METADATA:
        logger.warning(f"*** IN RELATION TO ['{TEST_ID}' EXPERIMENT, '{XAI_ALGORITHM}-{XAI_MODE}-{SURROGATE_MODEL}' XAI ALGORITHM, '{MASK_MODE}' MODE, '{MASK_RULE}' RULE, {MASK_RATE}' MASK RATE], THE MASKING PROCESS HAS ALREADY BEEN PERFORMED! ***\n")
        exit(1)
    
    DATASET = EXP_METADATA["DATASET"]
    CLASSES = list(EXP_METADATA["CLASSES"].keys())
    EXP_DIR = EXP_METADATA[f"{XAI_ALGORITHM}_{XAI_MODE}_{SURROGATE_MODEL}_METADATA"]["DIR_NAME"]
    PATCH_WIDTH = EXP_METADATA[f"{XAI_ALGORITHM}_{XAI_MODE}_{SURROGATE_MODEL}_METADATA"]["PATCH_DIM"]["WIDTH"]
    PATCH_HEIGHT = EXP_METADATA[f"{XAI_ALGORITHM}_{XAI_MODE}_{SURROGATE_MODEL}_METADATA"]["PATCH_DIM"]["HEIGHT"]
    
    instances, paths = setup_masking_process(DATASET, CLASSES, INSTANCE_SET)
    
    masker = None
    
    if MASK_MODE == "area":
        if MASK_MODE == "saliency":
            masker = AreaSaliencyMasker(test_id=TEST_ID, inst_set=INSTANCE_SET, instances=instances, paths=paths, 
            exp_dir=EXP_DIR, mask_rate=MASK_RATE, mask_rule=MASK_RULE, patch_width=PATCH_WIDTH, patch_height=PATCH_HEIGHT,
            xai_algorithm=XAI_ALGORITHM, xai_mode=XAI_MODE, surrogate_model=SURROGATE_MODEL, logger=logger, save_patches=True, verbose=True)
        elif MASK_MODE == "random":
            masker = AreaRandomMasker(test_id=TEST_ID, inst_set=INSTANCE_SET, instances=instances, paths=paths, 
            exp_dir=EXP_DIR, mask_rate=MASK_RATE, mask_rule=MASK_RULE, patch_width=PATCH_WIDTH, patch_height=PATCH_HEIGHT,
            xai_algorithm=XAI_ALGORITHM, xai_mode=XAI_MODE, surrogate_model=SURROGATE_MODEL, logger=logger, save_patches=True, verbose=True)
    
    elif MASK_MODE == "patch_number":
        if MASK_MODE == "saliency":
            masker = PatchNumberSaliencyMasker(test_id=TEST_ID, inst_set=INSTANCE_SET, instances=instances, paths=paths, 
            exp_dir=EXP_DIR, mask_rate=MASK_RATE, mask_rule=MASK_RULE, patch_width=PATCH_WIDTH, patch_height=PATCH_HEIGHT,
            xai_algorithm=XAI_ALGORITHM, xai_mode=XAI_MODE, surrogate_model=SURROGATE_MODEL, logger=logger, save_patches=True, verbose=True)
        elif MASK_MODE == "random":
            masker = PatchNumberRandomMasker(test_id=TEST_ID, inst_set=INSTANCE_SET, instances=instances, paths=paths, 
            exp_dir=EXP_DIR, mask_rate=MASK_RATE, mask_rule=MASK_RULE, patch_width=PATCH_WIDTH, patch_height=PATCH_HEIGHT,
            xai_algorithm=XAI_ALGORITHM, xai_mode=XAI_MODE, surrogate_model=SURROGATE_MODEL, logger=logger, save_patches=True, verbose=True)
    
    masker()