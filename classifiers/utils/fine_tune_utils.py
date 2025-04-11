import os, multiprocessing, random, torch
from torchvision.transforms import v2
from PIL import Image

from utils import get_train_instance_patterns, get_test_instance_patterns

from classifiers.utils.dataloader_utils import Test_DataLoader, all_crops, Invert, AddGaussianNoise
from classifiers.utils.testing_utils import produce_classification_reports 
from classifiers.classifier_ResNet18.model import load_resnet18_classifier

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
    if model_type == "ResNet18": model, last_cp = load_resnet18_classifier(num_classes, mode, cp_base, phase, test_id, exp_metadata, device, logger)
    
    logger.info("...Model successfully loaded!")

    return model, last_cp

def test_fine_tuned_model(test_id, exp_metadata, model_type, crop_size, OUTPUT_DIR, CLASSES_DATA, CP_BASE, mean_, std_, logger):
    exp_dir = f"{CLASSIFIERS_ROOT}/classifier_{model_type}/tests/{test_id}"
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_model(model_type, len(CLASSES_DATA), "frozen", CP_BASE, "test", test_id, exp_metadata, device, logger)
        
    n_crops_per_test_instance = exp_metadata["FINE_TUNING_HP"]["n_crops_per_test_instance"]
    dl = Test_DataLoader(directory=f"{exp_dir}/test", classes=list(CLASSES_DATA.keys()), batch_size=n_crops_per_test_instance, img_crop_size=crop_size, mean=mean_, std=std_)
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
        dst = f"{CLASSIFIERS_ROOT}/classifier_{model_type}/tests/{test_id}/train/{c}/{crop}"
        os.system(f"cp {src} {dst}")

### ################# ###
### CROP AUGMENTATION ###
### ################# ###
def augment_train_crops_parallel(exp_dir, class_name, mean_, std_):
    base_crops = os.listdir(f"{exp_dir}/train_pre_aug/{class_name}")
    base_crop_paths = [f"{exp_dir}/train_pre_aug/{class_name}/{crop}" for crop in base_crops]

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(apply_image_augmentations, [(crop_path, exp_dir, class_name, mean_, std_) for crop_path in base_crop_paths])
    
def apply_image_augmentations(args):
    crop_path, exp_dir, class_name, mean_, std_ = args
    crop = Image.open(crop_path)

    mean_int = tuple(int(m * 255) for m in mean_)
    
    cjitter = {'brightness': [0.4, 1.3], 'contrast': 0.6, 'saturation': 0.6, 'hue': (-0.4, 0.4)}
    randaffine = {'degrees': [-10,10], 'translate': [0.2, 0.2], 'scale': [1.3, 1.4], 'shear': 1, 'interpolation': v2.InterpolationMode.BILINEAR, 'fill': mean_int}
    randpersp = {'distortion_scale': 0.1, 'p': 0.2, 'interpolation': v2.InterpolationMode.BILINEAR, 'fill': mean_int}
    gray_p = 0.2
    gaussian_blur = {'kernel_size': 3, 'sigma': [0.1, 0.5]}
    # rand_eras = {'p': 0.5, 'scale': [0.02, 0.33], 'ratio': [0.3, 3.3]}
    # rand_eras['value'] = mean_
    invert_p = 0.05
    gaussian_noise = {'mean': 0., 'std': 0.004}
    gn_p = 0.0

    transforms = v2.Compose([
        v2.RandomAffine(**randaffine),
        v2.RandomPerspective(**randpersp),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianBlur(**gaussian_blur),
        v2.ColorJitter(**cjitter),
        v2.RandomGrayscale(gray_p),
        v2.RandomApply([Invert()], p=invert_p),
        v2.RandomApply([AddGaussianNoise(**gaussian_noise)], p=gn_p),
        v2.ToPILImage()
    ])

    crop_name = crop_path.split('/')[-1]
    augmented_crop = transforms(crop)
    augmented_crop.save(f"{exp_dir}/train/{class_name}/{crop_name}")