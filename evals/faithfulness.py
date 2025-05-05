import os, torch
import numpy as np
import pickle as pkl
from argparse import ArgumentParser
from datetime import datetime

from utils import load_metadata, save_metadata, get_model_base_checkpoint, get_logger, str2bool

from classifiers.utils.fine_tune_utils import load_model

from xai import EXPLAINERS, SURROGATES, MASK_MODES

from evals.utils.faithfulness_utils import get_test_instances_to_mask, mask_test_instances, test_model, produce_faithfulness_comparison_plot

LOG_ROOT = "./log"
EVAL_ROOT = "./evals"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=EXPLAINERS)
    parser.add_argument("-xai_mode", type=str, default="base")
    parser.add_argument("-surrogate_model", type=str, required=True, choices=SURROGATES)
    parser.add_argument("-mask_ceil", type=float, required=True)
    parser.add_argument("-mask_step", type=float, required=True)
    parser.add_argument("-mask_mode", type=str, required=True, choices=MASK_MODES)
    parser.add_argument("-keep_test_sets", type=str2bool, default=False)
    
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
    MASK_CEIL, MASK_STEP, MASK_MODE = args.mask_ceil, args.mask_step, args.mask_mode
    KEEP_TEST_SETS = args.keep_test_sets
    
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]
    DATASET = EXP_METADATA["DATASET"]
    CLASSES = list(EXP_METADATA["CLASSES"].keys())
    EXP_DIR = EXP_METADATA[f"{XAI_ALGORITHM}_{XAI_MODE}_{SURROGATE_MODEL}_METADATA"]["DIR_NAME"]
    PATCH_WIDTH = EXP_METADATA[f"{XAI_ALGORITHM}_{XAI_MODE}_{SURROGATE_MODEL}_METADATA"]["PATCH_DIM"]["WIDTH"]
    PATCH_HEIGHT = EXP_METADATA[f"{XAI_ALGORITHM}_{XAI_MODE}_{SURROGATE_MODEL}_METADATA"]["PATCH_DIM"]["HEIGHT"]
    
    CP_BASE = get_model_base_checkpoint(MODEL_TYPE)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    instances, paths = get_test_instances_to_mask(DATASET, CLASSES)
    
    current_mask_rate, mask_rates = 0.0, list()
    while current_mask_rate <= MASK_CEIL:
        current_mask_rate = round(current_mask_rate, 5)
        mask_rates.append(current_mask_rate)
        current_mask_rate += MASK_STEP
    
    performances = list()
    
    exp_eval_directory = f"{EVAL_ROOT}/faithfulness/{TEST_ID}/{XAI_ALGORITHM}_{XAI_MODE}_{SURROGATE_MODEL}"
    os.makedirs(exp_eval_directory, exist_ok=True)
    with open(f"{exp_eval_directory}/faithfulness_{MASK_MODE}_ceil{float(MASK_CEIL)*100}_step{float(MASK_STEP)*100}.txt", 'w') as f: 
        f.write(f"*** FAITHFULNESS COMPUTATION FOR TEST: {TEST_ID} ***\n")
    
    model, _ = load_model(MODEL_TYPE, len(CLASSES), "frozen", CP_BASE, "test", TEST_ID, None, DEVICE, logger)
    
    for i, mask_rate in enumerate(mask_rates):
        logger.info(f"BEGINNING OF ACCURACY COMPUTATION FOR M_RATE: {mask_rate}")
        if mask_rate == 0.0:
            os.makedirs(f"{exp_eval_directory}/test_set_masked_{MASK_MODE}_{mask_rate}", exist_ok=True)
            for inst, src in zip(instances, paths):
                c = src.split('/')[-2]
                os.makedirs(f"{exp_eval_directory}/test_set_masked_{MASK_MODE}_{mask_rate}/{c}", exist_ok=True)
                dest = f"{exp_eval_directory}/test_set_masked_{MASK_MODE}_{mask_rate}/{c}/{inst}"
                os.system(f"cp {src} {dest}")
        
        else: mask_test_instances(instances, paths, TEST_ID, EXP_DIR, mask_rate, MASK_MODE, PATCH_WIDTH, PATCH_HEIGHT, XAI_ALGORITHM, XAI_MODE, SURROGATE_MODEL, EXP_METADATA, logger)
        
        mask_rate_performances = test_model(model, DEVICE, CLASSES, EXP_METADATA, mask_rate, MASK_MODE, exp_eval_directory, logger)
        logger.info(f"TEST ACCURACY FOR M_RATE {mask_rate}: {mask_rate_performances}\n")
        if not KEEP_TEST_SETS:
            os.system(f"rm -rf {exp_eval_directory}/test_set_masked_{MASK_MODE}_{mask_rate}")
            
        with open(f"{exp_eval_directory}/faithfulness_{MASK_MODE}_ceil{float(MASK_CEIL)*100}_step{float(MASK_STEP)*100}.txt", 'a') as f:
            f.write(f"M_RATE: {mask_rate}; TEST ACCURACY: {mask_rate_performances}\n")
            f.write("### ------------------ ###\n\n")
            
        performances.append(mask_rate_performances)
    
    with open(f"{exp_eval_directory}/faithfulness_{MASK_MODE}_ceil{float(MASK_CEIL)*100}_step{float(MASK_STEP)*100}.pkl", 'wb') as f:
        pkl.dump(performances, f)
    
    performances = np.array(performances)
    performances = (performances - performances[0]) * -MASK_STEP
    
    faithfulness = np.sum(performances) / (len(mask_rates) * MASK_STEP)
    
    logger.info(f"Faithfulness for {TEST_ID}: {faithfulness}")
    
    with open(f"{exp_eval_directory}/faithfulness_{MASK_MODE}_ceil{float(MASK_CEIL)*100}_step{float(MASK_STEP)*100}.txt", 'a') as f:
        f.write("### ------------------ ###\n")
        f.write(f"Faithfulness: {faithfulness}")
    
    faithfulness_saliency_path = f"{exp_eval_directory}/faithfulness_saliency_ceil{float(MASK_CEIL)*100}_step{float(MASK_STEP)*100}.pkl"
    faithfulness_random_path = f"{exp_eval_directory}/faithfulness_random_ceil{float(MASK_CEIL)*100}_step{float(MASK_STEP)*100}.pkl"
    
    if os.path.exists(faithfulness_saliency_path) and os.path.exists(faithfulness_random_path):
        produce_faithfulness_comparison_plot(MASK_STEP, MASK_CEIL, exp_eval_directory)
    
    if "faithfulness_evals" not in EXP_METADATA[f"{XAI_ALGORITHM}_{XAI_MODE}_{SURROGATE_MODEL}_METADATA"]:
        EXP_METADATA[f"{XAI_ALGORITHM}_{XAI_MODE}_{SURROGATE_MODEL}_METADATA"]["faithfulness_evals"] = dict()
    EXP_METADATA[f"{XAI_ALGORITHM}_{XAI_MODE}_{SURROGATE_MODEL}_METADATA"]["faithfulness_evals"][f"{MASK_MODE}_ceil{float(MASK_CEIL)*100}_step{float(MASK_STEP)*100}"] = str(datetime.now())
    save_metadata(EXP_METADATA, EXP_METADATA_PATH)