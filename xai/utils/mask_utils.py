import os
from typing import Tuple

from utils import get_train_instance_patterns, get_test_instance_patterns

DATASET_ROOT = "./datasets"

def setup_masking_process(dataset, classes, instance_set) -> Tuple[list, list]:
    dataset_dir = f"{DATASET_ROOT}/{dataset}"
    instances, instance_full_paths = list(), list()
    
    patterns = get_train_instance_patterns() if instance_set == "train" else get_test_instance_patterns()
    pattern_func = patterns.get(dataset, lambda f: True)
    
    for c in classes:
        class_instances = [inst for inst in os.listdir(f"{dataset_dir}/{c}") if pattern_func(inst)]
        
        for inst in class_instances:
            instances.append(inst)
            instance_full_paths.append(f"{dataset_dir}/{c}/{inst}")
    
    return instances, instance_full_paths