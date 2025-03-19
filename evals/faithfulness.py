import os, torch
import numpy as np
import pickle as pkl
from argparse import ArgumentParser

from utils import load_metadata, get_model_base_checkpoint

from classifiers.utils.fine_tune_utils import load_model

from evals.utils.faithfulness_utils import get_test_instances_to_mask, mask_test_instances, test_model, produce_faithfulness_comparison_report

LOG_ROOT = "./log"
EVAL_ROOT = "./evals"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase"])
    parser.add_argument("-mask_ceil", type=float, required=True)
    parser.add_argument("-mask_step", type=float, required=True)
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
    MASK_CEIL, MASK_STEP, MASK_MODE = args.mask_ceil, args.mask_step, args.mask_mode
    
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]
    DATASET = EXP_METADATA["DATASET"]
    CLASSES = list(EXP_METADATA["CLASSES"].keys())
    EXP_DIR = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["DIR_NAME"]
    PATCH_WIDTH = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["PATCH_DIM"]["WIDTH"]
    PATCH_HEIGHT = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["PATCH_DIM"]["HEIGHT"]
    
    CP_BASE = get_model_base_checkpoint(MODEL_TYPE)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    instances, paths = get_test_instances_to_mask(DATASET, CLASSES)
    
    current_mask_rate, mask_rates = 0.0, list()
    while current_mask_rate <= MASK_CEIL:
        current_mask_rate = round(current_mask_rate, 5)
        mask_rates.append(current_mask_rate)
        current_mask_rate += MASK_STEP
    
    performances = list()
    
    os.makedirs(f"{EVAL_ROOT}/faithfulness/{TEST_ID}", exist_ok=True)
    with open(f"{EVAL_ROOT}/faithfulness/{TEST_ID}/faithfulness_{MASK_MODE}.txt", 'w') as f: f.write(f"*** FAITHFULNESS COMPUTATION FOR TEST: {TEST_ID} ***\n")
    
    model, _ = load_model(MODEL_TYPE, len(CLASSES), "frozen", CP_BASE, "test", TEST_ID, None, DEVICE)
    
    for i, mask_rate in enumerate(mask_rates):
        print(f"BEGINNING OF ACCURACY COMPUTATION FOR M_RATE: {mask_rate}")
        if mask_rate == 0.0:
            os.makedirs(f"{EVAL_ROOT}/faithfulness/{TEST_ID}/test_set_masked_{MASK_MODE}_{mask_rate}_{XAI_ALGORITHM}", exist_ok=True)
            for inst, src in zip(instances, paths):
                c = src.split('/')[-2]
                os.makedirs(f"{EVAL_ROOT}/faithfulness/{TEST_ID}/test_set_masked_{MASK_MODE}_{mask_rate}_{XAI_ALGORITHM}/{c}", exist_ok=True)
                dest = f"{EVAL_ROOT}/faithfulness/{TEST_ID}/test_set_masked_{MASK_MODE}_{mask_rate}_{XAI_ALGORITHM}/{c}/{inst}"
                os.system(f"cp {src} {dest}")
        
        else: mask_test_instances(instances, DATASET, paths, TEST_ID, EXP_DIR, mask_rate, MASK_MODE, PATCH_WIDTH, PATCH_HEIGHT, XAI_ALGORITHM, EXP_METADATA)
        
        mask_rate_performances = test_model(model, DEVICE, TEST_ID, CLASSES, EXP_METADATA, XAI_ALGORITHM, mask_rate, MASK_MODE)
        print(f"TEST ACCURACY FOR M_RATE {mask_rate}: {mask_rate_performances}\n")
            
        with open(f"{EVAL_ROOT}/faithfulness/{TEST_ID}/faithfulness_{MASK_MODE}.txt", 'a') as f:
            f.write(f"M_RATE: {mask_rate}; TEST ACCURACY: {mask_rate_performances}\n")
            f.write("### ------------------ ###\n\n")
            
        performances.append(mask_rate_performances)
    
    with open(f"{EVAL_ROOT}/faithfulness/{TEST_ID}/faithfulness_{MASK_MODE}.pkl", 'wb') as f:
        pkl.dump(performances, f)
    
    performances = np.array(performances)
    performances = (performances - performances[0]) * -MASK_STEP
    
    faithfulness = np.sum(performances) / (len(mask_rates) * MASK_STEP)
    
    print(f"Faithfulness for {TEST_ID}: {faithfulness}")
    
    with open(f"{EVAL_ROOT}/faithfulness/{TEST_ID}/faithfulness_{MASK_MODE}.txt", 'a') as f:
        f.write("### ------------------ ###\n")
        f.write(f"Faithfulness: {faithfulness}")
    
    faithfulness_saliency_path = f"{EVAL_ROOT}/faithfulness/{TEST_ID}/faithfulness_saliency.pkl"
    faithfulness_random_path = f"{EVAL_ROOT}/faithfulness/{TEST_ID}/faithfulness_random.pkl"
    
    if os.path.exists(faithfulness_saliency_path) and os.path.exists(faithfulness_random_path):
        produce_faithfulness_comparison_report(TEST_ID, MASK_STEP, MASK_CEIL)