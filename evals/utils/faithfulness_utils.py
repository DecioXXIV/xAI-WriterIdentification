import os
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from utils import get_test_instance_patterns

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

def mask_test_instances(instances, paths, test_id, exp_dir, mask_rate, mask_mode, patch_width, patch_height, xai_algorithm, xai_mode, surrogate_model, exp_metadata, logger):
    masker = None
    
    if mask_mode == "saliency":
        masker = SaliencyMasker(test_id=test_id, inst_set="test", instances=instances, paths=paths,
            exp_dir=exp_dir, mask_rate=mask_rate, mask_mode=mask_mode, patch_width=patch_width, 
            patch_height=patch_height, xai_algorithm=xai_algorithm, xai_mode=xai_mode, surrogate_model=surrogate_model,
            logger=logger, save_patches=False, verbose=True)
    elif mask_mode == "random":
        masker = RandomMasker(test_id=test_id, inst_set="test", instances=instances, paths=paths,
            exp_dir=exp_dir, mask_rate=mask_rate, mask_mode=mask_mode, patch_width=patch_width, 
            patch_height=patch_height, xai_algorithm=xai_algorithm, xai_mode=xai_mode, surrogate_model=surrogate_model,
            logger=logger, save_patches=False, verbose=False)
    
    masker()

def test_model(model, device, classes, exp_metadata, mask_rate, mask_mode, exp_eval_directory):
    test_set_dir = f"{exp_eval_directory}/test_set_masked_{mask_mode}_{mask_rate}"
    crop_size = exp_metadata["FINE_TUNING_HP"]["crop_size"]
    mean_, std_ = exp_metadata["FINE_TUNING_HP"]["mean"], exp_metadata["FINE_TUNING_HP"]["std"]
    
    dl = Eval_Test_DataLoader(test_set_dir, classes, 1, crop_size, 2, mean_, std_)
    
    _, labels, preds, _, idx_to_c = process_test_set(dl, device, model)
    label_class_names = [idx_to_c[id_] for id_ in labels]
    pred_class_names = [idx_to_c[id_] for id_ in preds]
    
    return accuracy_score(label_class_names, pred_class_names)

def produce_faithfulness_comparison_plot(mask_step, mask_ceil, exp_eval_directory):
    saliency_test_accs_path = f"{exp_eval_directory}/faithfulness_saliency_ceil{float(mask_ceil)*100}_step{float(mask_step)*100}.pkl"
    random_test_accs_path = f"{exp_eval_directory}/faithfulness_random_ceil{float(mask_ceil)*100}_step{float(mask_step)*100}.pkl"
    
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

    plt.savefig(f"{exp_eval_directory}/faithfulness_plot_ceil{float(mask_ceil)*100}_step{float(mask_step)*100}.png")