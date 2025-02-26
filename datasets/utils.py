import os
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

from utils import get_vert_hor_cuts, get_test_instance_patterns

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"

def split_full_instances(class_name, class_type, dataset, test_id, model_type, final_width, final_height):
    """
    Given a class and its 'class_type', processes and prepares its instances for the fine-tuning step.
    (final_width, final_height)-sized crops are extracted from each instance.

    Standard: final_width = 902, final_height = 1279

    Args:
        class_name (str): Name of the class
        class_type (str): Type of the class, 'base' or 'masked_saliency/random_maskrate_xaialgorithm' (i.e.: 'masked_saliency_0.1_LimeBase')
        dataset (str): Dataset name
        test_id (str): Unique test identifier
        model_type (str): Model type used in the experiment
        final_width (int): Desired width of the processed images (902)
        final_height (int): Desired height of the processed images (1279)
    """

    # Set 'source' and 'destination' for the current class, basing on its 'class_type'
    class_source, class_dest = None, None
    if class_type == "base": 
        class_source = f"{DATASET_ROOT}/{dataset}/{class_name}"
        class_dest = f"{DATASET_ROOT}/{dataset}/processed/{class_name}"
    else: 
        class_source = f"{DATASET_ROOT}/{dataset}/{class_name}-{test_id}_{model_type}_{class_type}"
        class_dest = f"{DATASET_ROOT}/{dataset}/processed/{class_name}-{test_id}_{model_type}_{class_type}"
    os.makedirs(class_dest, exist_ok=True)
    
    instances = os.listdir(class_source)
    
    for file in tqdm(instances, desc=f"Preparing Images of class {class_name}"):
        img = Image.open(f"{class_source}/{file}")
        img_width, img_height = img.size
        
        # Apply (symmetrical) padding, if necessary
        edited = False
        
        if (final_width > img_width) or (final_height > img_height):
            edited = True
            pad_left, pad_top, pad_right, pad_bottom = 0, 0, 0, 0
            
            if final_width > img_width:
                pad_left = (final_width - img_width) // 2
                pad_right = final_width - img_width - pad_left
            
            if final_height > img_height:
                pad_top = (final_height - img_height) // 2
                pad_bottom = final_height - img_height - pad_top
            
            transform = T.Pad((pad_left, pad_top, pad_right, pad_bottom), padding_mode="edge")
            img = transform(img)
            img_width, img_height = img.size
                    
        if edited: img.save(f"{class_source}/{file}")
        
        # Extract 'final_width'x'final_height' from the 'full' image
        vert_cuts, hor_cuts = get_vert_hor_cuts(dataset)
        h_overlap = max(1, int((((vert_cuts) * final_width) - img_width) / vert_cuts))
        v_overlap = max(1, int((((hor_cuts) * final_height) - img_height) / hor_cuts))
        
        for h_cut in range(0, hor_cuts):
            for v_cut in range(0, vert_cuts):
                left = v_cut * (final_width - h_overlap)
                right = left + final_width
                top = h_cut * (final_height - v_overlap)
                bottom = top + final_height
                
                subpage = img.crop((left, top, right, bottom))
                subpage.save(f"{class_dest}/{file[:-4]}_{h_cut}_{v_cut}{file[-4:]}")

def copy_not_masked_test_instances(class_name, class_type, dataset, test_id, model_type):
    """
    Copies 'test' instances that are not masked into the 'processed' dataset directory.
    This function is specific to "ret" experiments, where only 'train' instances are masked.

    Args:
        class_name (str): Name of the class.
        class_type (str): Type of the class.
        dataset (str): Dataset name.
        test_id (str): Unique test identifier.
        model_type (str): Model type used in the experiment.
    
    """

    base_id, _ = test_id.split(':')
    # In "ret" Experiments, the 'test_id' is composed by two parts
        # base_id -> Test Name (eg: CEDAR-Letter-1)
        # ret_id -> Re-Training specifications (eg: ret0.1_saliency_all)
    
    class_source = f"{DATASET_ROOT}/{dataset}/{class_name}" 
    class_dest = f"{DATASET_ROOT}/{dataset}/{class_name}-{base_id}_{model_type}_{class_type}"
    
    os.makedirs(class_dest, exist_ok=True)
    
    test_instance_patterns = get_test_instance_patterns()
    test_instances = [f for f in os.listdir(class_source) if test_instance_patterns[dataset](f)]
    
    for f in test_instances: os.system(f"cp {class_source}/{f} {class_dest}/{f}")