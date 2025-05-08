import pickle, os, json
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from PIL import Image
from copy import deepcopy
from torchvision import transforms as T

from utils import get_vert_hor_cuts, save_metadata, load_metadata, get_page_dimensions

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
XAI_ROOT = "./xai"
EVAL_ROOT = "./evals"

### Superclass ###
class ImageMasker:    
    def __init__(self, test_id: str, inst_set: str, instances: list, paths: list, 
                exp_dir: str, mask_rate: float, mask_mode: str,
                patch_width: int, patch_height: int, xai_algorithm: str, 
                xai_mode: str, surrogate_model: str, logger, save_patches=True, verbose=False):
        self.test_id = test_id
        self.inst_set = inst_set
        self.instances = instances
        self.paths = paths
        self.exp_dir = exp_dir
        self.mask_rate = mask_rate
        self.mask_mode = mask_mode
        self.patch_width, self.patch_height = patch_width, patch_height
        self.xai_algorithm = xai_algorithm
        self.xai_mode = xai_mode
        self.surrogate_model = surrogate_model
        self.logger = logger
        self.save_patches = save_patches
        self.verbose = verbose
        
        self.exp_metadata = load_metadata(f"{LOG_ROOT}/{self.test_id}-metadata.json", self.logger)
        self.dataset = self.exp_metadata["DATASET"]
        
        self.full_img_width, self.full_img_height = 0, 0
        self.final_width, self.final_height = get_page_dimensions(self.dataset)
        self.v_overlap, self.h_overlap = 0, 0
        
        self.training_mean = self.exp_metadata["FINE_TUNING_HP"]["mean"]
    
    def find_patches_coordinates(self, instance_name) -> list:
        """
        Given the 'instance_name', calculates the coordinates of patches to be masked.
        Returns a list of patches: each of them with their respective coordinates and score.
        """
        instance_directory = f"{XAI_ROOT}/explanations/patches_{self.patch_width}x{self.patch_height}_removal/{self.exp_dir}/{instance_name}"
        with open(f"{instance_directory}/padding_dict.json", "r") as f: padding_dict = json.load(f)

        filename_parts = instance_name.split("_")
        h_cut, v_cut = int(filename_parts[1]), int(filename_parts[2])
        
        left_b = v_cut * (self.final_width - self.h_overlap)
        top_b = h_cut * (self.final_height - self.v_overlap)
        
        crop_size = self.exp_metadata["FINE_TUNING_HP"]["crop_size"]
        overlap = self.exp_metadata[f"{self.xai_algorithm}_{self.xai_mode}_{self.surrogate_model}_METADATA"]["OVERLAP"]

        mask_array = np.load(f"{XAI_ROOT}/masks/{self.dataset}_mask_{self.patch_width}x{self.patch_height}_cs{crop_size}_overlap{overlap}/mask.npy")
        mask_height, mask_width = mask_array.shape
        mask_array[padding_dict["top"]:mask_height-padding_dict["bottom"], padding_dict["left"]:mask_width-padding_dict["right"]]
        
        with open(f"{XAI_ROOT}/explanations/patches_{self.patch_width}x{self.patch_height}_removal/{self.exp_dir}/{instance_name}/{instance_name}_scores.json", "r") as f: 
            scores = json.load(f)

        results = list()

        for idx, score in scores.items():
            instance_patch = f"{instance_name}_patch{idx}"
            
            positions = np.argwhere(mask_array == idx)
            if positions.size > 0:
                top_b_patch, left_b_patch = positions.min(axis=0)
                bottom_b_patch, right_b_patch = positions.max(axis=0)
                
                left_b_patch += left_b
                top_b_patch += top_b
                right_b_patch += left_b
                bottom_b_patch += top_b
                  
                results.append([instance_patch, score, left_b_patch, top_b_patch, right_b_patch, bottom_b_patch])
        
        self.logger.info(f"Patch Coordinates found for Instance: {instance_name}")
        return results
    
    def __call__(self):
        self.logger.info(f"*** Experiment: {self.test_id} -> BEGINNING OF MASKING PROCESS ***")
        self.logger.info(f"*** MODE = {self.mask_mode}, RATE = {self.mask_rate} ***")
        
        MODEL_TYPE = self.exp_metadata["MODEL_TYPE"]
        # EXP_METADATA_PATH = f"{LOG_ROOT}/{self.test_id}-metadata.json"
                
        for inst, path in zip(self.instances, self.paths):
            _ = self.mask_full_instance(inst, path)
            
            inst_name, inst_filetype = inst[:-4], inst[-4:]
            c = int(inst_name[0:4])
                        
            src_path = f"{self.INSTANCE_DIR_MODE_RATE}/{inst_name}_masked_{self.mask_mode}_{self.mask_rate}{inst_filetype}"

            # 'train' Instances are masked to be then used for "ret" Experiments: 
            if self.inst_set == "train": dest_dir = f"{DATASET_ROOT}/{self.dataset}/{c}-{self.test_id}_{MODEL_TYPE}_masked_{self.mask_mode}_{self.mask_rate}_{self.xai_algorithm}"

            # 'test' Instances are masked to be then used for the "Faithfulness" evaluation
            elif self.inst_set == "test": dest_dir = f"{EVAL_ROOT}/faithfulness/{self.test_id}/{self.xai_algorithm}_{self.xai_mode}_{self.surrogate_model}/test_set_masked_{self.mask_mode}_{self.mask_rate}/{c}"

            os.makedirs(dest_dir, exist_ok=True)
            os.system(f"cp {src_path} {dest_dir}/{inst_name}{inst_filetype}")
            
        self.logger.info(f"*** Experiment: {self.test_id} -> END OF MASKING PROCESS FOR TEST ***\n")
        
        # self.exp_metadata[f"MASK_PROCESS_{self.inst_set}_{self.mask_mode}_{self.mask_rate}_{self.xai_algorithm}_{self.xai_mode}_END_TIMESTAMP"] = str(datetime.now())
        # save_metadata(self.exp_metadata, EXP_METADATA_PATH)

### Subclasses ###
class SaliencyMasker(ImageMasker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def mask_full_instance(self, full_img_name, full_img_path): 
        """This method performs the Saliency masking for a given instance."""
        if self.verbose: self.logger.info(f"Processing Saliency Masking for Instance: {full_img_name}")
        full_img = Image.open(full_img_path)
        
        self.full_img_width, self.full_img_height = full_img.size
        full_img_name, full_img_type = full_img_name[:-4], full_img_name[-4:]
        
        vert_cuts, hor_cuts = get_vert_hor_cuts(self.exp_metadata["DATASET"])
        self.h_overlap = max(1, int((((vert_cuts) * self.final_width) - self.full_img_width) / vert_cuts))
        self.v_overlap = max(1, int((((hor_cuts) * self.final_height) - self.full_img_height) / hor_cuts))  
        
        ROOT_OF_MASKINGS = f"{XAI_ROOT}/masked_images/{self.exp_dir}"
        self.INSTANCE_DIR = f"{ROOT_OF_MASKINGS}/{full_img_name}"
        os.makedirs(self.INSTANCE_DIR, exist_ok=True)
        
        MASK_METADATA_PATH = f"{ROOT_OF_MASKINGS}/{self.inst_set}_{self.mask_mode}_{self.mask_rate}_{self.xai_algorithm}-metadata.json"
        MASK_METADATA = None
        if not os.path.exists(MASK_METADATA_PATH):
            MASK_METADATA = {"INSTANCES": dict()}
            save_metadata(MASK_METADATA, MASK_METADATA_PATH)
        else:
            MASK_METADATA = load_metadata(MASK_METADATA_PATH, self.logger)
        
        if not os.path.exists(f"{self.INSTANCE_DIR}/masking_results.csv"):
            if self.verbose: self.logger.info("PHASE 1 -> PATCHES MAPPING")
            directory = f"{XAI_ROOT}/explanations/patches_{self.patch_width}x{self.patch_height}_removal/{self.exp_dir}"
            instances = [i for i in os.listdir(directory) if full_img_name in i]
            
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(self.find_patches_coordinates, instances)
            
            all_results = [item for sublist in results for item in sublist]
            df = pd.DataFrame(all_results, columns=["Instance_Patch", "Score", "Coordinates_Left", "Coordinates_Top", "Coordinates_Right", "Coordinates_Bottom"])
            df = df.sort_values(by="Score", ascending=False)
            df.to_csv(f"{self.INSTANCE_DIR}/masking_results.csv", index=False, header=True)
        
        else:
            if self.verbose: self.logger.warning("PHASE 1 SKIPPED -> PATCHES MAPPING ALREADY AVAILABLE")
            df = pd.read_csv(f"{self.INSTANCE_DIR}/masking_results.csv", header=0)
        
        if self.verbose: self.logger.info("PHASE 2 -> MASKING PROCESS")
        
        self.INSTANCE_DIR_MODE_RATE = f"{self.INSTANCE_DIR}/{self.mask_mode}_{self.mask_rate}"
        os.makedirs(self.INSTANCE_DIR_MODE_RATE, exist_ok=True)
        
        if self.save_patches and not os.path.exists(f"{self.INSTANCE_DIR}/removed_patches"):
            os.makedirs(f"{self.INSTANCE_DIR}/removed_patches", exist_ok=True)
            for row in range(0, len(df)):
                patch_id = df.iloc[row]["Instance_Patch"].split('_')[-1]
                score = df.iloc[row]["Score"]
                left = df.iloc[row]["Coordinates_Left"]
                top = df.iloc[row]["Coordinates_Top"] 
                right = df.iloc[row]["Coordinates_Right"] 
                bottom = df.iloc[row]["Coordinates_Bottom"]
                
                patch = full_img.crop((left, top, right+1, bottom+1))
                patch.save(f"{self.INSTANCE_DIR}/removed_patches/{row+1}_{patch_id}_{score}.jpg")
        
        full_img_area = self.full_img_width * self.full_img_height   
        full_img_to_mask = deepcopy(full_img)
        full_img_to_mask_tensor = T.ToTensor()(full_img_to_mask)

        """
        Saliency Masking Process:
        Patches are sorted by decreasing Attribution Score and masking is performed
        one patch at a time, until the required "mask_rate" is reached. 

        Saliency: masking is stopped if the currently examined patch is related to
        a not-positive Attribution Score.
        """
        
        check_array = np.ones(shape=(self.full_img_height, self.full_img_width))
        masked_patches, masked_area, row = 0, 0, 0
        end_condition = False
        
        while not end_condition:
            patch_id = df.iloc[row]["Instance_Patch"].split('_')[-1]
            left = df.iloc[row]["Coordinates_Left"]
            top = df.iloc[row]["Coordinates_Top"] 
            right = df.iloc[row]["Coordinates_Right"] 
            bottom = df.iloc[row]["Coordinates_Bottom"]
                    
            for channel, mean_v in enumerate(self.training_mean):
                full_img_to_mask_tensor[channel, top:bottom+1, left:right+1] = mean_v
            check_array[top:bottom+1, left:right+1] = 0
            masked_patches += 1
            row += 1
                
            masked_area = np.count_nonzero(check_array == 0)
            if (masked_area >= full_img_area * self.mask_rate):
                end_condition = True
                    
        full_img_masked_tensor_copy = deepcopy(full_img_to_mask_tensor)
        full_img_masked = T.ToPILImage()(full_img_masked_tensor_copy)
        full_img_masked.save(f"{self.INSTANCE_DIR_MODE_RATE}/{full_img_name}_masked_{self.mask_mode}_{self.mask_rate}{full_img_type}")
        
        if self.verbose: self.logger.info(f"End of Masking Process for the current Instance -> Masked Patches: {masked_patches}\n")
        
        masked_area_ratio = masked_area / full_img_area
        MASK_METADATA["INSTANCES"][f"{full_img_name}"] = masked_area_ratio
        save_metadata(MASK_METADATA, MASK_METADATA_PATH)
        
        return masked_area_ratio
        
class RandomMasker(ImageMasker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def mask_full_instance(self, full_img_name, full_img_path):
        """
        This method performs the Random masking for a given instance.
        """
        if self.verbose: self.logger.info(f"Processing Random Masking for Instance: {full_img_name}")
        full_img = Image.open(full_img_path)
        
        self.full_img_width, self.full_img_height = full_img.size
        full_img_name, full_img_type = full_img_name[:-4], full_img_name[-4:]
        
        vert_cuts, hor_cuts = get_vert_hor_cuts(self.exp_metadata["DATASET"])
        self.h_overlap = max(1, int((((vert_cuts) * self.final_width) - self.full_img_width) / vert_cuts))
        self.v_overlap = max(1, int((((hor_cuts) * self.final_height) - self.full_img_height) / hor_cuts)) 
        
        ROOT_OF_MASKINGS = f"{XAI_ROOT}/masked_images/{self.exp_dir}"
        self.INSTANCE_DIR = f"{ROOT_OF_MASKINGS}/{full_img_name}"
        os.makedirs(self.INSTANCE_DIR, exist_ok=True)
        
        MASK_METADATA_PATH = f"{ROOT_OF_MASKINGS}/{self.inst_set}_{self.mask_mode}_{self.mask_rate}_{self.xai_algorithm}-metadata.json"
        MASK_METADATA = None
        if not os.path.exists(MASK_METADATA_PATH):
            MASK_METADATA = {"INSTANCES": dict()}
            save_metadata(MASK_METADATA, MASK_METADATA_PATH)
        else:
            MASK_METADATA = load_metadata(MASK_METADATA_PATH, self.logger)
        
        if self.verbose: self.logger.info("MASKING PROCESS")
        
        self.INSTANCE_DIR_MODE_RATE = f"{self.INSTANCE_DIR}/{self.mask_mode}_{self.mask_rate}"
        os.makedirs(self.INSTANCE_DIR_MODE_RATE, exist_ok=True)
        
        full_img_area = self.full_img_width * self.full_img_height   
        full_img_to_mask = deepcopy(full_img)
        full_img_to_mask_tensor = T.ToTensor()(full_img_to_mask)

        """
        Random Masking Process:
        Masking is performed by drawing one (patch_width, patch_height)-sized patch at a time.

        Drawn patches do not follow the the rigid structure defined by the Mask.
        """
        
        check_array = np.ones(shape=(self.full_img_height, self.full_img_width))
        masked_patches, masked_area, idx  = 0, 0, 0
        
        end_condition = False
        while not end_condition:
            left = np.random.randint(0, self.full_img_width - self.patch_width + 1)
            right = left + self.patch_width - 1
            top = np.random.randint(0, self.full_img_height - self.patch_height + 1)
            bottom = top + self.patch_height - 1
                
            for channel, mean_v in enumerate(self.training_mean):
                full_img_to_mask_tensor[channel, top:bottom+1, left:right+1] = mean_v
                
            check_array[top:bottom+1, left:right+1] = 0
            idx, masked_patches = idx + 1, masked_patches + 1
                
            masked_area = np.count_nonzero(check_array == 0)
            if (masked_area > full_img_area * self.mask_rate):
                end_condition = True
                    
        full_img_masked_tensor_copy = deepcopy(full_img_to_mask_tensor)
        full_img_masked = T.ToPILImage()(full_img_masked_tensor_copy)
        full_img_masked.save(f"{self.INSTANCE_DIR_MODE_RATE}/{full_img_name}_masked_{self.mask_mode}_{self.mask_rate}{full_img_type}")
        
        if self.verbose: self.logger.info(f"End of Masking Process for the current Instance -> Masked Patches: {masked_patches}\n")
        
        masked_area_ratio = masked_area / full_img_area
        MASK_METADATA["INSTANCES"][f"{full_img_name}"] = masked_area_ratio
        save_metadata(MASK_METADATA, MASK_METADATA_PATH)
        
        return masked_area_ratio