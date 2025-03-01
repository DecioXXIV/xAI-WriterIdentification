import os, torch
from sklearn.metrics import accuracy_score

from utils import load_metadata, get_test_instance_patterns, get_model_base_checkpoint

from xai.maskers.image_masker import SaliencyMasker, RandomMasker

from classifiers.utils.fine_tune_utils import load_model
from classifiers.utils.dataloader_utils import Faithfulness_Test_DataLoader, load_rgb_mean_std
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

def mask_test_instances(instances, paths, test_id, exp_dir, mask_rate, mask_mode, patch_width, patch_height, xai_algorithm, exp_metadata):
    masker = None
    
    if mask_mode == "saliency":
        masker = SaliencyMasker(inst_set="test", instances=instances, paths=paths, test_id=test_id,
            exp_dir=exp_dir, mask_rate=mask_rate, mask_mode=mask_mode,
            patch_width=patch_width, patch_height=patch_height,
            xai_algorithm=xai_algorithm, xai_mode="base", exp_metadata=exp_metadata)
    elif mask_mode == "random":
        masker = RandomMasker(inst_set="test", instances=instances, paths=paths, test_id=test_id,
            exp_dir=exp_dir, mask_rate=mask_rate, mask_mode=mask_mode,
            patch_width=patch_width, patch_height=patch_height,
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

def test_model(model_type, test_id, classes, xai_algorithm, mask_rate, mask_mode):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CP_BASE = get_model_base_checkpoint(model_type)
    
    model, _ = load_model(model_type, len(classes), "frozen", CP_BASE, "test", test_id, None, DEVICE)
    
    EXP_DIR = f"{CLASSIFIERS_ROOT}/classifier_{model_type}/tests/{test_id}"
    
    test_set_dir = f"{EVAL_ROOT}/faithfulness/{test_id}/test_set_masked_{mask_mode}_{mask_rate}_{xai_algorithm}"
    mean_, std_ = load_rgb_mean_std(f"{EXP_DIR}")
    
    dl = Faithfulness_Test_DataLoader(directory=test_set_dir, classes=classes, batch_size=1, img_crop_size=380, mean=mean_, std=std_)
    
    _, labels, preds, _, idx_to_c = process_test_set(dl, DEVICE, model)
    label_class_names = [idx_to_c[id_] for id_ in labels]
    pred_class_names = [idx_to_c[id_] for id_ in preds]
    
    return accuracy_score(label_class_names, pred_class_names)