import os, torch
import numpy as np
from argparse import ArgumentParser

from utils import load_metadata

from evals.utils.faithfulness_utils import get_test_instances_to_mask, mask_test_instances, test_model

LOG_ROOT = "./log"
EVAL_ROOT = "./evals"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase"])
    parser.add_argument("-mask_ceil", type=float, required=True)
    parser.add_argument("-mask_step", type=float, required=True)
    parser.add_argument("-mask_mode", type=str, required=True, choices=["saliency", "random"])
    parser.add_argument("-test_iterations", type=int, default=1)
    
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
    TEST_ITERATIONS = args.test_iterations
    
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]
    DATASET = EXP_METADATA["DATASET"]
    CLASSES = list(EXP_METADATA["CLASSES"].keys())
    EXP_DIR = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["DIR_NAME"]
    PATCH_WIDTH = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["PATCH_DIM"]["WIDTH"]
    PATCH_HEIGHT = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["PATCH_DIM"]["HEIGHT"]
    
    instances, paths = get_test_instances_to_mask(DATASET, CLASSES)
    
    mask_rates = np.arange(start=0.0, stop=MASK_CEIL+MASK_STEP, step=MASK_STEP)
    performances = list()
    
    os.makedirs(f"{EVAL_ROOT}/faithfulness/{TEST_ID}", exist_ok=True)
    with open(f"{EVAL_ROOT}/faithfulness/{TEST_ID}/faithfulness_{MASK_MODE}.txt", 'w') as f: f.write(f"*** FAITHFULNESS COMPUTATION FOR TEST: {TEST_ID} ***\n")
    
    for i, mask_rate in enumerate(mask_rates):
        print(f"BEGINNING OF ACCURACY COMPUTATION FOR M_RATE: {mask_rate}")
        if mask_rate == 0.0:
            os.makedirs(f"{EVAL_ROOT}/faithfulness/{TEST_ID}/test_set_masked_{MASK_MODE}_{mask_rate}_{XAI_ALGORITHM}", exist_ok=True)
            for inst, src in zip(instances, paths):
                c = src.split('/')[-2]
                os.makedirs(f"{EVAL_ROOT}/faithfulness/{TEST_ID}/test_set_masked_{MASK_MODE}_{mask_rate}_{XAI_ALGORITHM}/{c}", exist_ok=True)
                dest = f"{EVAL_ROOT}/faithfulness/{TEST_ID}/test_set_masked_{MASK_MODE}_{mask_rate}_{XAI_ALGORITHM}/{c}/{inst}"
                os.system(f"cp {src} {dest}")
            mask_condition = True
        
        else: mask_condition = mask_test_instances(instances, paths, TEST_ID, EXP_DIR, mask_rate, MASK_MODE, PATCH_WIDTH, PATCH_HEIGHT, XAI_ALGORITHM, EXP_METADATA)
        
        if mask_condition:
            mask_rate_performances = np.zeros(TEST_ITERATIONS)
            for iter in range(0, TEST_ITERATIONS):
                mask_rate_performances[iter] = test_model(MODEL_TYPE, TEST_ID, CLASSES, XAI_ALGORITHM, mask_rate, MASK_MODE)
                torch.cuda.empty_cache()
            
            mean_perf = np.mean(mask_rate_performances)
            var_perf = np.var(mask_rate_performances)
            print(f"MEAN ACCURACY ON {TEST_ITERATIONS} ITERATIONS FOR M_RATE {mask_rate}: {mean_perf}\n")
            
            with open(f"{EVAL_ROOT}/faithfulness/{TEST_ID}/faithfulness_{MASK_MODE}.txt", 'a') as f:
                f.write(f"M_RATE: {mask_rate}; ITERATIONS: {TEST_ITERATIONS}\n")
                f.write(f"Mean Accuracy: {mean_perf}; Variance: {var_perf}\n")
                f.write("### ------------------ ###\n\n")
            
            performances.append(mean_perf)
        else:
            print(f"Faithfulness Computation stopped on m_rate: {mask_rate}")
            break
    
    performances = np.array(performances)
    performances = (performances - performances[0]) * -MASK_STEP
    
    faithfulness = np.sum(performances) / (len(mask_rates) * MASK_STEP)
    
    print(f"Faithfulness for {TEST_ID}: {faithfulness}")
    
    with open(f"{EVAL_ROOT}/faithfulness/{TEST_ID}/faithfulness_{MASK_MODE}.txt", 'a') as f:
        f.write("### ------------------ ###\n")
        f.write(f"Faithfulness: {faithfulness}")