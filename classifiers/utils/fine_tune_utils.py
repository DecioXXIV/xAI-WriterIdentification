import os, multiprocessing, random, torch
from PIL import Image

from utils import get_train_instance_patterns, get_test_instance_patterns

from classifiers.utils.dataloader_utils import Test_DataLoader, all_crops
from classifiers.utils.testing_utils import produce_classification_reports 
from classifiers.classifier_ResNet18.model import load_resnet18_classifier
from classifiers.classifier_ViT_SwinTiny.model import load_vit_swintiny_classifier

LOG_ROOT = "./log"
CLASSIFIERS_ROOT = "./classifiers"
XAI_AUG_ROOT = "./xai_augmentation"

def create_directories(root_folder, classes):
    os.makedirs(root_folder, exist_ok=True)
    
    for phase in ["train", "val", "test"]:
        for c in classes:
            os.makedirs(f"{root_folder}/{phase}/{c}", exist_ok=True)
    
    os.makedirs(f"{root_folder}/output", exist_ok=True)

def load_model(model_type, num_classes, mode, cp_base, phase, test_id, exp_metadata, device, logger):
    logger.info(f"Loading Model '{model_type}'...")
    model, last_cp = None, None
    if model_type == "ResNet18": 
        model, last_cp = load_resnet18_classifier(num_classes, mode, cp_base, phase, test_id, exp_metadata, device, logger)
    elif model_type == "ViT_SwinTiny":
        model, last_cp = load_vit_swintiny_classifier(num_classes, phase, test_id, exp_metadata, device, logger)
    
    logger.info("...Model successfully loaded!")

    return model, last_cp

def test_fine_tuned_model(test_id, exp_metadata, model_type, crop_size, OUTPUT_DIR, CLASSES_DATA, CP_BASE, mean_, std_, logger):
    exp_dir = f"{CLASSIFIERS_ROOT}/classifier_{model_type}/tests/{test_id}"
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_model(model_type, len(CLASSES_DATA), "frozen", CP_BASE, "test", test_id, exp_metadata, device, logger)
        
    n_crops_per_test_instance = exp_metadata["FINE_TUNING_HP"]["n_crops_per_test_instance"]
    dl = Test_DataLoader(model_type=model_type, directory=f"{exp_dir}/test", classes=list(CLASSES_DATA.keys()), batch_size=n_crops_per_test_instance, img_crop_size=crop_size, mean=mean_, std=std_)
    produce_classification_reports(dl, device, model, OUTPUT_DIR, test_id)

### ############### ###
### CROP EXTRACTION ###
### ############### ###
def process_file_new(args):
    file, phase, exp_dir, source_dir, class_name, replicas, crop_size, mult_factor = args
    
    img = Image.open(os.path.join(source_dir, file))
    crops = all_crops(img, (crop_size, crop_size), mult_factor)
    
    if phase == "train":
        os.makedirs(f"{exp_dir}/train_pre_aug/{class_name}", exist_ok=True)
        for i in range(replicas):
            for n, crop in enumerate(crops): crop.save(f"{exp_dir}/train_pre_aug/{class_name}/{file[:-4]}_cp{i+1}_crop{n+1}{file[-4:]}")
        
    elif phase == "test":
        test_crop_names = [f"{file[:-4]}_crop{n+1}{file[-4:]}" for n in range(len(crops))]
        for n, crop in enumerate(crops): crop.save(f"{exp_dir}/test/{class_name}/{test_crop_names[n]}")
        
        n_val_crops = int(len(crops)/4)
        val_crop_names = random.sample(test_crop_names, n_val_crops)
        val_crops = [crops[test_crop_names.index(c)] for c in val_crop_names]
        for n, crop in enumerate(val_crops): crop.save(f"{exp_dir}/val/{class_name}/{val_crop_names[n]}")
    
    return len(crops)

def extract_crops_parallel(dataset, exp_dir, source_dir, class_name, train_replicas, crop_size, train_dl_mf):
    files = os.listdir(source_dir)
    
    train = [f for f in files if get_train_instance_patterns()[dataset](f)]
    test = [f for f in files if get_test_instance_patterns()[dataset](f)]
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)

    with multiprocessing.Pool(num_workers) as pool:
        train_crops = pool.map(process_file_new, 
                               [(file, "train", exp_dir, source_dir, class_name, train_replicas, crop_size, train_dl_mf) 
                                for file in train])
    
    with multiprocessing.Pool(num_workers) as pool:
        test_crops = pool.map(process_file_new, 
                              [(file, "test", exp_dir, source_dir, class_name, train_replicas, crop_size, train_dl_mf*2) 
                               for file in test])
    
    # "pool.map()" returns a list of values, one for each file processed
    n_crops_per_train_instance, n_crops_per_test_instance = train_crops[0], test_crops[0]
    return n_crops_per_train_instance, n_crops_per_test_instance

def retrieve_augmentation_crops(test_id, model_type, c):
    base_id, aug_id = test_id.split(':')
    aug_id_parts = aug_id.split('_')
    aug_mode, balance = aug_id_parts[-3], aug_id_parts[-2]
    
    augmented_crops = os.listdir(f"{XAI_AUG_ROOT}/{base_id}/{aug_mode}_{balance}/crops_for_augmentation/{c}")
    for crop in augmented_crops:
        src = f"{XAI_AUG_ROOT}/{base_id}/{aug_mode}_{balance}/crops_for_augmentation/{c}/{crop}"
        dst = f"{CLASSIFIERS_ROOT}/classifier_{model_type}/tests/{test_id}/train_pre_aug/{c}/{crop}"
        os.system(f"cp {src} {dst}")