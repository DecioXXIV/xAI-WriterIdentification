import os, multiprocessing
from PIL import Image

from utils import get_train_instance_patterns, get_test_instance_patterns
from classifiers.utils.dataloader_utils import all_crops, n_random_crops
from classifiers.classifier_ResNet18.model import load_resnet18_classifier

CLASSIFIERS_ROOT = "./classifiers"
XAI_AUG_ROOT = "./xai_augmentation"

def create_directories(root_folder, classes):
    os.makedirs(root_folder, exist_ok=True)
    
    for phase in ["train", "val", "test"]:
        for c in classes:
            os.makedirs(f"{root_folder}/{phase}/{c}", exist_ok=True)
    
    os.makedirs(f"{root_folder}/output", exist_ok=True)

### ############### ###
### CROP EXTRACTION ###
### ############### ###
def process_train_file(args):
    file, exp_dir, source_dir, class_name, train_replicas, crop_size, mult_factor = args
    
    img = Image.open(os.path.join(source_dir, file))
    crops = all_crops(img, (crop_size, crop_size), mult_factor)
    
    for i in range(train_replicas):
        for n, crop in enumerate(crops): crop.save(f"{exp_dir}/train/{class_name}/{file[:-4]}_cp{i+1}_crop{n+1}{file[-4:]}")

def process_test_file(args):
    file, exp_dir, source_dir, class_name, crop_size, test_n_crops, random_seed = args
    
    img = Image.open(os.path.join(source_dir, file))
    
    val_crops = n_random_crops(img, int(test_n_crops/4), (crop_size, crop_size), random_seed)
    for n, crop in enumerate(val_crops): crop.save(f"{exp_dir}/val/{class_name}/{file[:-4]}_crop{n+1}{file[-4:]}")
    
    test_crops = n_random_crops(img, test_n_crops, (crop_size, crop_size), random_seed)
    for n, crop in enumerate(test_crops): crop.save(f"{exp_dir}/test/{class_name}/{file[:-4]}_crop{n+1}{file[-4:]}")

def extract_crops_parallel(dataset, exp_dir, source_dir, class_name, train_replicas, crop_size, mult_factor, test_n_crops, random_seed):
    files = os.listdir(source_dir)
    
    train_instance_patterns = get_train_instance_patterns()
    test_instance_patterns = get_test_instance_patterns()
    
    train = [f for f in files if train_instance_patterns[dataset](f)]
    test = [f for f in files if test_instance_patterns[dataset](f)]
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(process_train_file, [(file, exp_dir, source_dir, class_name, train_replicas, crop_size, mult_factor) for file in train])
    
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(process_test_file, [(file, exp_dir, source_dir, class_name, crop_size, test_n_crops, random_seed) for file in test])

def retrieve_augmentation_crops(test_id, model_type, c):
    base_id, aug_id = test_id.split(':')
    aug_id_parts = aug_id.split('_')
    aug_mode, balance = aug_id_parts[-3], aug_id_parts[-2]
    
    augmented_crops = os.listdir(f"{XAI_AUG_ROOT}/{base_id}/{aug_mode}_{balance}/crops_for_augmentation/{c}")
    for crop in augmented_crops:
        src = f"{XAI_AUG_ROOT}/{base_id}/{aug_mode}_{balance}/crops_for_augmentation/{c}/{crop}"
        dst = f"{CLASSIFIERS_ROOT}/classifier_{model_type}/tests/{test_id}/train/{c}/{crop}"
        os.system(f"cp {src} {dst}")
    
def load_model(model_type, num_classes, mode, cp_base, phase, test_id, exp_metadata, device):
    print(f"Loading Model '{model_type}'...")
    model, last_cp = None, None
    if model_type == "ResNet18":
        model, last_cp = load_resnet18_classifier(num_classes, mode, cp_base, phase, test_id, exp_metadata, device)
    
    print("...Model successfully loaded!")

    return model, last_cp