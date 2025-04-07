import os, torch
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from utils import load_metadata, get_test_instance_patterns

from xai.maskers.image_masker import SaliencyMasker, RandomMasker

from classifiers.utils.dataloader_utils import Eval_Test_DataLoader
from classifiers.utils.testing_utils import process_test_set

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
CLASSIFIERS_ROOT = "./classifiers"
XAI_ROOT = "./xai"
EVAL_ROOT = "./evals"

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

def mask_test_instances(instances, paths, test_id, exp_dir, mask_rate, mask_mode, patch_width, patch_height, xai_algorithm, surrogate_model, exp_metadata):
    masker = None
    
    if mask_mode == "saliency":
        masker = SaliencyMasker(test_id=test_id, inst_set="test", instances=instances, paths=paths,
            exp_dir=exp_dir, mask_rate=mask_rate, mask_mode=mask_mode, patch_width=patch_width, 
            patch_height=patch_height, xai_algorithm=xai_algorithm, xai_mode="base", surrogate_model=surrogate_model,
            save_patches=False, verbose=True)
    elif mask_mode == "random":
        masker = RandomMasker(test_id=test_id, inst_set="test", instances=instances, paths=paths,
            exp_dir=exp_dir, mask_rate=mask_rate, mask_mode=mask_mode, patch_width=patch_width, 
            patch_height=patch_height, xai_algorithm=xai_algorithm, xai_mode="base", surrogate_model=surrogate_model,
            save_patches=False, verbose=True)
    
    masker()
    
    dir_name = exp_metadata[f"{xai_algorithm}_base_{surrogate_model}_METADATA"]["DIR_NAME"]
    MASK_METADATA_PATH = f"{XAI_ROOT}/masked_images/{dir_name}/test_{mask_mode}_{mask_rate}_{xai_algorithm}-metadata.json"
    MASK_METADATA = load_metadata(MASK_METADATA_PATH)
    
    mask_rate2instance = MASK_METADATA["INSTANCES"]
    total_instances, bad_instances = len(mask_rate2instance), 0
    for _, masked_area in mask_rate2instance.items():
        if masked_area < mask_rate: bad_instances += 1
    
    if bad_instances < 0.33 * total_instances: return True
    else: return False

def test_model(model, device, test_id, classes, exp_metadata, xai_algorithm, mask_rate, mask_mode, exp_eval_directory):
    test_set_dir = f"{exp_eval_directory}/test_set_masked_{mask_mode}_{mask_rate}"
    crop_size = exp_metadata["FINE_TUNING_HP"]["crop_size"]
    mean_, std_ = exp_metadata["FINE_TUNING_HP"]["mean"], exp_metadata["FINE_TUNING_HP"]["std"]
    
    dl = Eval_Test_DataLoader(test_set_dir, classes, 1, crop_size, 2, mean_, std_)
    
    _, labels, preds, _, idx_to_c = process_test_set(dl, device, model)
    label_class_names = [idx_to_c[id_] for id_ in labels]
    pred_class_names = [idx_to_c[id_] for id_ in preds]
    
    return accuracy_score(label_class_names, pred_class_names)

def produce_faithfulness_comparison_report(test_id, mask_step, mask_ceil, exp_eval_directory):
    saliency_test_accs_path = f"{exp_eval_directory}/faithfulness_saliency.pkl"
    random_test_accs_path = f"{exp_eval_directory}/faithfulness_random.pkl"
    
    saliency_test_accs, random_test_accs = None, None
    with open(saliency_test_accs_path, "rb") as f: saliency_test_accs = pkl.load(f)
    with open(random_test_accs_path, "rb") as f: random_test_accs = pkl.load(f)
    
    current_mask_rate, mask_rates = 0.0, list()
    while current_mask_rate <= mask_ceil:
        current_mask_rate = round(current_mask_rate, 5)
        mask_rates.append(current_mask_rate)
        current_mask_rate += mask_step
    
    plt.figure(figsize=(10, 5))
    plt.plot(mask_rates, saliency_test_accs, marker='o', linestyle='-', color='b', label='Saliency Removals')
    plt.plot(mask_rates, random_test_accs, marker='s', linestyle='--', color='r', label='Random Removals')

    plt.xlabel("Mask Rate")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{exp_eval_directory}/faithfulness_plot.png")