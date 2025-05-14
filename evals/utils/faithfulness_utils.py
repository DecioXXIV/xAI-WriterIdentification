import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from utils import get_test_instance_patterns

from xai.maskers.area_image_masker import AreaSaliencyMasker, AreaRandomMasker
from xai.maskers.patch_number_image_masker import PatchNumberSaliencyMasker, PatchNumberRandomMasker

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

def mask_test_instances(instances, paths, test_id, exp_dir, mask_rate, mask_rule, mask_mode, patch_width, patch_height, xai_algorithm, xai_mode, surrogate_model, exp_metadata, logger):
    masker = None
    
    if mask_mode == "area":
        if mask_rule == "saliency":
            masker = AreaSaliencyMasker(test_id=test_id, inst_set="test", instances=instances, paths=paths,
            exp_dir=exp_dir, mask_rate=mask_rate, mask_rule=mask_rule, patch_width=patch_width,
            patch_height=patch_height, xai_algorithm=xai_algorithm, xai_mode=xai_mode, surrogate_model=surrogate_model,
            logger=logger, save_patches=False, verbose=True)
        
        elif mask_rule == "random":
            masker = AreaRandomMasker(test_id=test_id, inst_set="test", instances=instances, paths=paths,
            exp_dir=exp_dir, mask_rate=mask_rate, mask_rule=mask_rule, patch_width=patch_width,
            patch_height=patch_height, xai_algorithm=xai_algorithm, xai_mode=xai_mode, surrogate_model=surrogate_model,
            logger=logger, save_patches=False, verbose=False)
    
    elif mask_mode == "patch_number":
        if mask_rule == "saliency":
            masker = PatchNumberSaliencyMasker(test_id=test_id, inst_set="test", instances=instances, paths=paths,
            exp_dir=exp_dir, mask_rate=mask_rate, mask_rule=mask_rule, patch_width=patch_width,
            patch_height=patch_height, xai_algorithm=xai_algorithm, xai_mode=xai_mode, surrogate_model=surrogate_model,
            logger=logger, save_patches=False, verbose=True)
        
        elif mask_rule == "random":
            masker = PatchNumberRandomMasker(test_id=test_id, inst_set="test", instances=instances, paths=paths,
            exp_dir=exp_dir, mask_rate=mask_rate, mask_rule=mask_rule, patch_width=patch_width,
            patch_height=patch_height, xai_algorithm=xai_algorithm, xai_mode=xai_mode, surrogate_model=surrogate_model,
            logger=logger, save_patches=False, verbose=False)

    masker()

def test_model(model, model_type, device, classes, exp_metadata, mask_rate, mask_rule, mask_mode, exp_eval_directory, logger):
    test_set_dir = f"{exp_eval_directory}/test_set_masked_{mask_mode}_{mask_rule}_{mask_rate}"
    crop_size = exp_metadata["FINE_TUNING_HP"]["crop_size"]
    mean_, std_ = exp_metadata["FINE_TUNING_HP"]["mean"], exp_metadata["FINE_TUNING_HP"]["std"]
    
    dl = Eval_Test_DataLoader(model_type, test_set_dir, classes, 1, crop_size, 2, mean_, std_)
    
    logger.info(f"Test Accuracy Evaluation for '{mask_mode}' Masking, {mask_rule} Rule and '{mask_rate}' Mask Rate")
    _, labels, preds, _, idx_to_c = process_test_set(dl, device, model)
    label_class_names = [idx_to_c[id_] for id_ in labels]
    pred_class_names = [idx_to_c[id_] for id_ in preds]
    
    return accuracy_score(label_class_names, pred_class_names)

def compute_final_faithfulness_value(test_id, raw_performances, exp_eval_directory, mask_rates, mask_rule, mask_mode, mask_ceil, mask_step, logger):
    if mask_rule == "saliency":
        performances = np.array(raw_performances)
    
    elif mask_rule == "random":
        performances = np.zeros(len(mask_rates))
        for i in range(0, len(mask_rates)): performances[i] = float(np.mean(raw_performances[i]))
    
    performances = (performances - performances[0]) * -mask_step
    faithfulness = np.sum(performances) / (len(mask_rates) * mask_step)
    
    logger.info(f"Faithfulness for {test_id}: {faithfulness}")
    
    with open(f"{exp_eval_directory}/faithfulness_{mask_rule}_{mask_mode}_ceil{float(mask_ceil)*100}_step{float(mask_step)*100}.txt", 'a') as f:
        f.write("### ------------------ ###\n")
        f.write(f"Faithfulness: {faithfulness}")
    
def produce_faithfulness_comparison_plot(exp_eval_directory, mask_ceil, mask_step, mask_mode, mask_rates):
    # Saliency Test Accuracies
    saliency_test_accs_path = f"{exp_eval_directory}/faithfulness_saliency_{mask_mode}_ceil{float(mask_ceil)*100}_step{float(mask_step)*100}.pkl"
    with open(saliency_test_accs_path, "rb") as f: saliency_test_accs = pkl.load(f)
    
    # Random Test Accuracies
    random_test_accs_path = f"{exp_eval_directory}/faithfulness_random_{mask_mode}_ceil{float(mask_ceil)*100}_step{float(mask_step)*100}.pkl"
    with open(random_test_accs_path, "rb") as f: random_test_accs = pkl.load(f)
    iters = random_test_accs.shape[1]
    
    random_mean_test_accs, random_max_test_accs, random_min_test_accs = list(), list(), list()
    for i in range(0, len(mask_rates)):
        random_mean_test_accs.append(float(np.mean(random_test_accs[i])))
        random_max_test_accs.append(float(np.max(random_test_accs[i])))
        random_min_test_accs.append(float(np.min(random_test_accs[i])))
    
    plt.figure(figsize=(10, 5))
    plt.plot(mask_rates, saliency_test_accs, marker='o', linestyle='-', color='b', label="Saliency Removals")
    plt.plot(mask_rates, random_mean_test_accs, marker='s', linestyle='--', color='r', label="Random Removals (mean)")
    plt.fill_between(mask_rates, random_min_test_accs, random_max_test_accs, color='r', alpha=0.2, label=f"Random Removals Range ({iters} iters, min-to-max)")
    
    plt.xlabel("Mask Rate")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{exp_eval_directory}/faithfulness_plot_{mask_mode}_ceil{float(mask_ceil)*100}_step{float(mask_step)*100}.png")