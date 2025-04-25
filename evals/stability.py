import os, json
from argparse import ArgumentParser

from utils import load_metadata, get_logger

from evals.utils.stability_utils import get_instances, get_faith_evaluated_xai_modes, collect_all_instances_attr_scores_per_patch_parallel, get_mode_pairs, produce_instances_abs_differences_report_parallel, compute_global_abs_differences_stats, produce_instances_jaccard_report_parallel

LOG_ROOT = "./log"
DATASETS_ROOT = "./datasets"
XAI_ROOT = "./xai"
EVAL_ROOT = "./evals"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase", "GLimeBinomial"])
    parser.add_argument("-surrogate_model", type=str, required=True, choices=["Ridge"])
    parser.add_argument("-xai_modes", type=str, default=None)
    parser.add_argument("-stability_eval_name", type=str, required=True)

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
    
    XAI_ALGORITHM, SURROGATE_MODEL = args.xai_algorithm, args.surrogate_model
    XAI_MODES = args.xai_modes
    STABILITY_EVAL_NAME = args.stability_eval_name
    
    DATASET = EXP_METADATA["DATASET"]
    CLASSES = list(EXP_METADATA["CLASSES"].keys())
    
    logger.info(f"*** BEGINNING OF STABILITY EVALUATION FOR ['{TEST_ID}', '{XAI_ALGORITHM}', '{SURROGATE_MODEL}'] ***")
    
    os.makedirs(f"{EVAL_ROOT}/stability/{TEST_ID}", exist_ok=True)
    instances = get_instances(DATASET, CLASSES)
    for inst in instances: os.makedirs(f"{EVAL_ROOT}/stability/{TEST_ID}/{STABILITY_EVAL_NAME}/instances/{inst}", exist_ok=True)
    
    logger.info("Producing Reports on Absolute Patch Attribution Scores Differences for all instances...")
    
    modes_mapping = get_faith_evaluated_xai_modes(TEST_ID, XAI_ALGORITHM, SURROGATE_MODEL, XAI_MODES, logger)
    with open(f"{EVAL_ROOT}/stability/{TEST_ID}/{STABILITY_EVAL_NAME}/modes_mapping.json", 'w') as jf: json.dump(modes_mapping, jf, indent=4)
    
    collect_all_instances_attr_scores_per_patch_parallel(TEST_ID, instances, modes_mapping, EXP_METADATA, STABILITY_EVAL_NAME)
    logger.info("Collected all instances' attribute scores per patch")
    
    mode_pairs = get_mode_pairs(modes_mapping)
    produce_instances_abs_differences_report_parallel(TEST_ID, instances, modes_mapping, mode_pairs, STABILITY_EVAL_NAME)
    logger.info("Produced absolute differences reports for all instances")
    
    compute_global_abs_differences_stats(TEST_ID, instances, STABILITY_EVAL_NAME)
    
    logger.info("Producing Reports on Jaccard Similarity for all instances...")
    produce_instances_jaccard_report_parallel(TEST_ID, instances, modes_mapping, mode_pairs, STABILITY_EVAL_NAME, logger)
    logger.info(f"*** END OF STABILITY EVALUATION FOR ['{TEST_ID}', '{XAI_ALGORITHM}', '{SURROGATE_MODEL}'] ***")