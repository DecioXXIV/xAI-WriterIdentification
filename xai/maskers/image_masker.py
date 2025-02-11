import pickle, os
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from PIL import Image
from copy import deepcopy
from torchvision import transforms as T

from xai.utils.explanations_utils import reduce_scores

from utils import get_vert_hor_cuts, save_metadata, load_metadata

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
XAI_ROOT = "./xai"
EVAL_ROOT = "./evals"

### Superclass ###
class ImageMasker:    
    def __init__(self, inst_set: str, instances: list, paths: list, test_id: str,
                exp_dir: str, mask_rate: float, mask_mode: str,
                block_width: int, block_height: int,
                xai_algorithm: str, xai_mode: str, exp_metadata: dict):
        self.inst_set = inst_set
        self.instances = instances
        self.paths = paths
        self.test_id = test_id
        self.exp_dir = exp_dir
        self.mask_rate = mask_rate
        self.mask_mode = mask_mode
        self.block_width, self.block_height = block_width, block_height
        self.xai_algorithm = xai_algorithm
        self.xai_mode = xai_mode
        self.exp_metadata = exp_metadata
        
        self.full_img_width, self.full_img_height = 0, 0
        self.v_overlap, self.h_overlap = 0, 0
        
        with open(f"{XAI_ROOT}/explanations/patches_{self.block_width}x{self.block_height}_removal/{exp_dir}/rgb_train_stats.pkl", "rb") as f:
            self.training_mean, _ = pickle.load(f)
    
    def find_patches_coordinates(self, instance_name) -> list:
        """
        Given the 'instance_name', calculates the coordinates of patches to be masked.
        Returns a list of patches: each of them with their respective coordinates and score.
        """

        filename_parts = instance_name.split("_")
        h_cut, v_cut = int(filename_parts[1]), int(filename_parts[2])
        
        left_b = v_cut * (902 - self.h_overlap)
        top_b = h_cut * (1279 - self.v_overlap)

        mask = Image.open(f"{XAI_ROOT}/def_mask_{self.block_width}x{self.block_height}.png")
        mask_array = np.array(mask)
        
        with open(f"{XAI_ROOT}/explanations/patches_{self.block_width}x{self.block_height}_removal/{self.exp_dir}/{instance_name}/{instance_name}_scores.pkl", "rb") as f: 
            base_scores = pickle.load(f)
        reduced_scores = reduce_scores(mask, base_scores)

        results = list()

        for idx, score in reduced_scores.items():
            if score != [np.nan]:
                instance_patch = f"{instance_name}_block{idx}"
                
                positions = np.argwhere(mask_array == idx)
                if positions.size > 0:
                    top_b_patch, left_b_patch = positions.min(axis=0)
                    bottom_b_patch, right_b_patch = positions.max(axis=0)
                
                    left_b_patch += left_b
                    top_b_patch += top_b
                    right_b_patch += left_b
                    bottom_b_patch += top_b
                    
                    results.append([instance_patch, score, left_b_patch, top_b_patch, right_b_patch, bottom_b_patch])
        
        print(f"Processed Instance: {instance_name}")
        return results
    
    def __call__(self):
        print(f"*** BEGINNING OF MASKING PROCESS FOR TEST: {self.test_id} ***")
        print(f"*** MODE = {self.mask_mode}, RATE = {self.mask_rate} ***")
        
        DATASET = self.exp_metadata["DATASET"]
        TEST_ID = self.exp_metadata["TEST_ID"]
        MODEL_TYPE = self.exp_metadata["MODEL_TYPE"]
        EXP_METADATA_PATH = f"{LOG_ROOT}/{TEST_ID}-metadata.json"
        
        if f"MASK_PROCESS_{self.inst_set}_{self.mask_mode}_{self.mask_rate}_{self.xai_algorithm}_{self.xai_mode}_END_TIMESTAMP" in self.exp_metadata:
            print(f"*** IN RELATION TO THE EXPERIMENT '{self.test_id}' AND THE SETTING (xai_algorithm={self.xai_algorithm}-{self.xai_mode}, mode={self.mask_mode}, mask_rate={self.mask_rate}), THE INSTANCES HAVE ALREADY BEEN MASKED! ***")
            exit(1)
        
        # Masking Process
        else:
            for inst, path in zip(self.instances, self.paths): 
                _ = self.mask_full_instance(inst, path)
            
                inst_name, inst_type = inst[:-4], inst[-4:]
                c = int(inst_name[0:4])
                        
                src_path = f"{self.INSTANCE_DIR_MODE_RATE}/{inst_name}_masked_{self.mask_mode}_{self.mask_rate}{inst_type}"

                # 'train' Instances are masked to be then used for "ret" Experiments: 
                if self.inst_set == "train": dest_dir = f"{DATASET_ROOT}/{DATASET}/{c}-{TEST_ID}_{MODEL_TYPE}_masked_{self.mask_mode}_{self.mask_rate}_{self.xai_algorithm}"

                # 'test' Instances are masked to be then used for the "Faithfulness" evaluation
                elif self.inst_set == "test": dest_dir = f"{EVAL_ROOT}/faithfulness/{TEST_ID}/test_set_masked_{self.mask_mode}_{self.mask_rate}_{self.xai_algorithm}/{c}"

                os.makedirs(dest_dir, exist_ok=True)
                os.system(f"cp {src_path} {dest_dir}/{inst_name}{inst_type}")
            
            print(f"*** END OF MASKING PROCESS FOR TEST: {self.test_id} ***\n")
        
            self.exp_metadata[f"MASK_PROCESS_{self.inst_set}_{self.mask_mode}_{self.mask_rate}_{self.xai_algorithm}_{self.xai_mode}_END_TIMESTAMP"] = str(datetime.now())
            save_metadata(self.exp_metadata, EXP_METADATA_PATH)

### Subclasses ###
class SaliencyMasker(ImageMasker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def mask_full_instance(self, full_img_name, full_img_path):
        """
        This method performs the Saliency masking for a given instance.
        """

        print(f"Processing Saliency Masking for Instance: {full_img_name}")
        full_img = Image.open(full_img_path)
        
        self.full_img_width, self.full_img_height = full_img.size
        full_img_name, full_img_type = full_img_name[:-4], full_img_name[-4:]
        
        FINAL_WIDTH, FINAL_HEIGHT = 902, 1279
        vert_cuts, hor_cuts = get_vert_hor_cuts(self.exp_metadata["DATASET"])
        self.h_overlap = max(1, int((((vert_cuts) * FINAL_WIDTH) - self.full_img_width) / vert_cuts))
        self.v_overlap = max(1, int((((hor_cuts) * FINAL_HEIGHT) - self.full_img_height) / hor_cuts)) 
        
        ROOT_OF_MASKINGS = f"{XAI_ROOT}/mask_images/{self.exp_dir}"
        self.INSTANCE_DIR = f"{ROOT_OF_MASKINGS}/{full_img_name}"
        os.makedirs(self.INSTANCE_DIR, exist_ok=True)
        
        MASK_METADATA_PATH = f"{ROOT_OF_MASKINGS}/{self.inst_set}_{self.mask_mode}_{self.mask_rate}_{self.xai_algorithm}-metadata.json"
        MASK_METADATA = None
        if not os.path.exists(MASK_METADATA_PATH):
            MASK_METADATA = {"INSTANCES": dict()}
            save_metadata(MASK_METADATA, MASK_METADATA_PATH)
        else:
            MASK_METADATA = load_metadata(MASK_METADATA_PATH)
        
        if not os.path.exists(f"{self.INSTANCE_DIR}/masking_results.csv"):
            print("PHASE 1 -> PATCHES MAPPING")
            instances = [i for i in os.listdir(f"{XAI_ROOT}/explanations/patches_{self.block_width}x{self.block_height}_removal/{self.exp_dir}") if full_img_name in i]
            
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(self.find_patches_coordinates, instances)
            
            all_results = [item for sublist in results for item in sublist]
            df = pd.DataFrame(all_results, columns=["Instance_Block", "Score", "Coordinates_Left", "Coordinates_Top", "Coordinates_Right", "Coordinates_Bottom"])
            df = df.sort_values(by="Score", ascending=False)
            df.to_csv(f"{self.INSTANCE_DIR}/masking_results.csv", index=False, header=True)
        
        else:
            print("PHASE 1 SKIPPED -> PATCHES MAPPING ALREADY AVAILABLE")
            df = pd.read_csv(f"{self.INSTANCE_DIR}/masking_results.csv", header=0)

        print("PHASE 2 -> MASKING PROCESS")
        
        self.INSTANCE_DIR_MODE_RATE = f"{self.INSTANCE_DIR}/{self.mask_mode}_{self.mask_rate}"
        
        full_img_area = self.full_img_width * self.full_img_height   
        full_img_to_mask = deepcopy(full_img)
        full_img_to_mask_tensor = T.ToTensor()(full_img_to_mask)
        
        os.makedirs(f"{self.INSTANCE_DIR_MODE_RATE}/removed_patches", exist_ok=True)

        """
        Saliency Masking Process:
        Patches are sorted by decreasing Attribution Score and masking is performed
        one patch at a time, until the required "mask_rate" is reached. 

        Saliency: masking is stopped if the currently examined patch is related to
        a not-positive Attribution Score.
        """
        
        check_array = np.ones(shape=(self.full_img_width, self.full_img_height))
        masked_patches, masked_area, idx  = 0, 0, 0
        end_condition = False
        while not end_condition:
            left = df.iloc[idx]["Coordinates_Left"]
            top = df.iloc[idx]["Coordinates_Top"] 
            right = df.iloc[idx]["Coordinates_Right"] 
            bottom = df.iloc[idx]["Coordinates_Bottom"]
                    
            if (right - left + 1 != self.block_width) or (bottom - top + 1 != self.block_height):
                print(f"Skipped patch {idx} due to wrong dimensions")
                idx += 1
                continue
            
            if df.iloc[idx]["Score"] <= 0:
                end_condition = True
            
            else:
                for channel, mean_v in enumerate(self.training_mean):
                    full_img_to_mask_tensor[channel, top:bottom, left:right] = mean_v
                
                check_array[top:bottom, left:right] = 0
                idx, masked_patches = idx + 1, masked_patches + 1
                
                patch = full_img.crop((left, top, right, bottom))
                patch.save(f"{self.INSTANCE_DIR_MODE_RATE}/removed_patches/patch_{masked_patches}.jpg")
                
                masked_area = np.count_nonzero(check_array == 0)
                if (masked_area > full_img_area * self.mask_rate):
                    end_condition = True
                    
        full_img_masked_tensor_copy = deepcopy(full_img_to_mask_tensor)
        full_img_masked = T.ToPILImage()(full_img_masked_tensor_copy)
        full_img_masked.save(f"{self.INSTANCE_DIR_MODE_RATE}/{full_img_name}_masked_{self.mask_mode}_{self.mask_rate}{full_img_type}")
        
        print(f"End of Masking Process for the current Instance -> Masked Patches: {masked_patches}\n")
        
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
        print(f"Processing Random Masking for Instance: {full_img_name}")
        full_img = Image.open(full_img_path)
        
        self.full_img_width, self.full_img_height = full_img.size
        full_img_name, full_img_type = full_img_name[:-4], full_img_name[-4:]
        
        FINAL_WIDTH, FINAL_HEIGHT = 902, 1279
        vert_cuts, hor_cuts = get_vert_hor_cuts(self.exp_metadata["DATASET"])
        self.h_overlap = max(1, int((((vert_cuts) * FINAL_WIDTH) - self.full_img_width) / vert_cuts))
        self.v_overlap = max(1, int((((hor_cuts) * FINAL_HEIGHT) - self.full_img_height) / hor_cuts)) 
        
        ROOT_OF_MASKINGS = f"{XAI_ROOT}/mask_images/{self.exp_dir}"
        self.INSTANCE_DIR = f"{ROOT_OF_MASKINGS}/{full_img_name}"
        os.makedirs(self.INSTANCE_DIR, exist_ok=True)
        
        MASK_METADATA_PATH = f"{ROOT_OF_MASKINGS}/{self.inst_set}_{self.mask_mode}_{self.mask_rate}_{self.xai_algorithm}-metadata.json"
        MASK_METADATA = None
        if not os.path.exists(MASK_METADATA_PATH):
            MASK_METADATA = {"INSTANCES": dict()}
            save_metadata(MASK_METADATA, MASK_METADATA_PATH)
        else:
            MASK_METADATA = load_metadata(MASK_METADATA_PATH)
        
        if not os.path.exists(f"{self.INSTANCE_DIR}/masking_results.csv"):
            print("PHASE 1 -> PATCHES MAPPING")
            instances = [i for i in os.listdir(f"{XAI_ROOT}/explanations/patches_{self.block_width}x{self.block_height}_removal/{self.exp_dir}") if full_img_name in i]
            
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(self.find_patches_coordinates, instances)
            
            all_results = [item for sublist in results for item in sublist]
            df = pd.DataFrame(all_results, columns=["Instance_Block", "Score", "Coordinates_Left", "Coordinates_Top", "Coordinates_Right", "Coordinates_Bottom"])
            df = df.sort_values(by="Score", ascending=False)
            df.to_csv(f"{self.INSTANCE_DIR}/masking_results.csv", index=False, header=True)
        
        else:
            print("PHASE 1 SKIPPED -> PATCHES MAPPING ALREADY AVAILABLE")
            df = pd.read_csv(f"{self.INSTANCE_DIR}/masking_results.csv", header=0)

        print("PHASE 2 -> MASKING PROCESS")
        
        self.INSTANCE_DIR_MODE_RATE = f"{self.INSTANCE_DIR}/{self.mask_mode}_{self.mask_rate}"
        
        full_img_area = self.full_img_width * self.full_img_height   
        full_img_to_mask = deepcopy(full_img)
        full_img_to_mask_tensor = T.ToTensor()(full_img_to_mask)
        
        os.makedirs(f"{self.INSTANCE_DIR_MODE_RATE}/removed_patches", exist_ok=True)

        """
        Random Masking Process:
        Masking is performed by drawing one (block_width, block_height)-sized
        patch at a time.

        Random: drawn patches do not follow the the rigid structure defined
        by the Mask.
        """
        
        check_array = np.ones(shape=(self.full_img_width, self.full_img_height))
        masked_patches, masked_area, idx  = 0, 0, 0
        
        end_condition = False
        while not end_condition:
            left = np.random.randint(0, self.full_img_width - self.block_width + 1)
            right = left + self.block_width
            top = np.random.randint(0, self.full_img_height - self.block_height + 1)
            bottom = top + self.block_height
                    
            if (right - left != self.block_width) or (bottom - top != self.block_height):
                print(f"Skipped patch {idx} due to wrong dimensions")
                idx += 1
                continue
            
            else:
                for channel, mean_v in enumerate(self.training_mean):
                    full_img_to_mask_tensor[channel, top:bottom, left:right] = mean_v
                
                check_array[top:bottom, left:right] = 0
                idx, masked_patches = idx + 1, masked_patches + 1
                
                patch = full_img.crop((left, top, right, bottom))
                patch.save(f"{self.INSTANCE_DIR_MODE_RATE}/removed_patches/patch_{masked_patches}.jpg")
                
                masked_area = np.count_nonzero(check_array == 0)
                if (masked_area > full_img_area * self.mask_rate):
                    end_condition = True
                    
        full_img_masked_tensor_copy = deepcopy(full_img_to_mask_tensor)
        full_img_masked = T.ToPILImage()(full_img_masked_tensor_copy)
        full_img_masked.save(f"{self.INSTANCE_DIR_MODE_RATE}/{full_img_name}_masked_{self.mask_mode}_{self.mask_rate}{full_img_type}")
        
        print(f"End of Masking Process for the current Instance -> Masked Patches: {masked_patches}\n")
        
        masked_area_ratio = masked_area / full_img_area
        MASK_METADATA["INSTANCES"][f"{full_img_name}"] = masked_area_ratio
        save_metadata(MASK_METADATA, MASK_METADATA_PATH)
        
        return masked_area_ratio