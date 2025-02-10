import os, torch
import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score

from utils import load_metadata, get_test_instance_patterns

from xai.maskers.image_masker_new import SaliencyMasker, RandomMasker
from classifiers.classifier_NN.fine_tune import load_model
from classifiers.classifier_NN.utils.dataloader_utils import Train_Test_DataLoader, load_rgb_mean_std
from classifiers.classifier_NN.utils.testing_utils import process_test_set

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
CLASSIFIER_ROOT = "./classifiers/classifier_NN"
XAI_ROOT = "./xai"
EVAL_ROOT = "./evals"

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase"])
    parser.add_argument("-mask_ceil", type=float, required=True)
    parser.add_argument("-mask_step", type=float, required=True)
    parser.add_argument("-mask_mode", type=str, required=True, choices=["saliency", "random"])
    parser.add_argument("-test_iterations", type=int, default=1)
    
    return parser.parse_args()

def get_test_instances_to_mask(dataset, classes):
    dataset_dir = f"{DATASET_ROOT}/{dataset}"
    instances, instance_full_paths = list(), list()
    
    patterns = get_test_instance_patterns()
    pattern_func = patterns.get(dataset, lambda f: True)
    
    for c in classes:
        class_instances = [inst for inst in os.listdir(f"{dataset_dir}/{c}") if pattern_func(inst)]
        
        for inst in class_instances:
            instances.append(inst)
            instance_full_paths.append(f"{dataset_dir}/{c}/{inst}")
    
    return instances, instance_full_paths

def mask_test_instances(instances, paths, test_id, exp_dir, mask_rate, mask_mode, block_width, block_height, xai_algorithm, exp_metadata):
    masker = None
    
    if mask_mode == "saliency":
        masker = SaliencyMasker(inst_set="test", instances=instances, paths=paths, test_id=test_id,
            exp_dir=exp_dir, mask_rate=mask_rate, mask_mode=mask_mode,
            block_width=block_width, block_height=block_height,
            xai_algorithm=xai_algorithm, xai_mode="base", exp_metadata=exp_metadata)
    elif mask_mode == "random":
        masker = RandomMasker(inst_set="test", instances=instances, paths=paths, test_id=test_id,
            exp_dir=exp_dir, mask_rate=mask_rate, mask_mode=mask_mode,
            block_width=block_width, block_height=block_height,
            xai_algorithm=xai_algorithm, xai_mode="base", exp_metadata=exp_metadata)
    
    masker()
    
    dir_name = exp_metadata[f"{xai_algorithm}_base_METADATA"]["DIR_NAME"]
    MASK_METADATA_PATH = f"{XAI_ROOT}/mask_images/{dir_name}/test_{mask_mode}_{mask_rate}_{xai_algorithm}-metadata.json"
    MASK_METADATA = load_metadata(MASK_METADATA_PATH)
    
    mask_rate2instance = MASK_METADATA["INSTANCES"]
    total_instances, bad_instances = len(mask_rate2instance), 0
    for _, masked_area in mask_rate2instance.items():
        if masked_area < mask_rate: bad_instances += 1
    
    if bad_instances < 0.33 * total_instances: return True
    else: return False

def test_model(test_id, classes, xai_algorithm, mask_rate, mask_mode):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CP_BASE = f"{CLASSIFIER_ROOT}/../cp/Test_3_TL_val_best_model.pth"
    
    model, _ = load_model(len(classes), "frozen", CP_BASE, "test", test_id, None, DEVICE)
    EXP_DIR = f"{CLASSIFIER_ROOT}/tests/{test_id}"
    
    test_set_dir = f"{EVAL_ROOT}/faithfulness/{test_id}/test_set_masked_{mask_mode}_{mask_rate}_{xai_algorithm}"
    mean_, std_ = load_rgb_mean_std(f"{EXP_DIR}")
    
    dl = Train_Test_DataLoader(directory=test_set_dir, classes=classes, batch_size=1, img_crop_size=380, weighted_sampling=False, phase='test', mean=mean_, std=std_, shuffle=True)
    
    _, labels, preds, _, idx_to_c = process_test_set(dl, DEVICE, model)
    label_class_names = [idx_to_c[id_] for id_ in labels]
    pred_class_names = [idx_to_c[id_] for id_ in preds]
    
    return accuracy_score(label_class_names, pred_class_names)
    
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
    BLOCK_WIDTH = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["BLOCK_DIM"]["WIDTH"]
    BLOCK_HEIGHT = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["BLOCK_DIM"]["HEIGHT"]
    
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
        
        else: mask_condition = mask_test_instances(instances, paths, TEST_ID, EXP_DIR, mask_rate, MASK_MODE, BLOCK_WIDTH, BLOCK_HEIGHT, XAI_ALGORITHM, EXP_METADATA)
        
        if mask_condition:
            mask_rate_performances = np.zeros(TEST_ITERATIONS)
            for iter in range(0, TEST_ITERATIONS):
                mask_rate_performances[iter] = test_model(TEST_ID, CLASSES, XAI_ALGORITHM, mask_rate, MASK_MODE)
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