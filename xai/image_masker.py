import pickle, os, json
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from PIL import Image
from copy import deepcopy
from torchvision import transforms as T

from .explainers import reduce_scores

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
XAI_ROOT = "./xai"

class ImageMasker:    
    def __init__(self,
                instances: list,
                paths: list,
                test_id: str,
                exp_dir: str,
                mask_rate: float,
                mode: str,
                block_width: int,
                block_height: int,
                xai_algorithm: str,
                exp_metadata: dict):
        self.instances = instances
        self.paths = paths
        self.test_id = test_id
        self.exp_dir = exp_dir
        self.mask_rate = mask_rate
        self.mode = mode
        self.block_width, self.block_height = block_width, block_height
        self.xai_algorithm = xai_algorithm
        self.exp_metadata = exp_metadata
        
        self.full_img_width, self.full_img_height = 0, 0
        self.v_overlap, self.h_overlap = 0, 0
        
        with open(f"{XAI_ROOT}/explanations/patches_{self.block_width}x{self.block_height}_removal/{exp_dir}/rgb_train_stats.pkl", "rb") as f:
            self.training_mean, _ = pickle.load(f)
        
    def mask_full_instance(self, full_img_name, full_img_path):
        print(f"Processing Instance: {full_img_name}")
        full_img = Image.open(full_img_path)
        
        self.full_img_width, self.full_img_height = full_img.size
        full_img_name, full_img_type = full_img_name[:-4], full_img_name[-4:]
        
        FINAL_WIDTH, FINAL_HEIGHT = 902, 1279
        VERT_MULT_FACT, HOR_MULT_FACT = self.exp_metadata["PREP_MULT_FACT"]["VERT"], self.exp_metadata["PREP_MULT_FACT"]["HOR"]
        
        vert_cuts = (self.full_img_width // FINAL_WIDTH) * VERT_MULT_FACT
        hor_cuts = (self.full_img_height // FINAL_HEIGHT) * HOR_MULT_FACT
        
        self.h_overlap = max(1, int((((vert_cuts + 1) * FINAL_WIDTH) - self.full_img_width) / vert_cuts))
        self.v_overlap = max(1, int((((hor_cuts + 1) * FINAL_HEIGHT) - self.full_img_height) / hor_cuts)) 
        
        ROOT_OF_MASKINGS = f"{XAI_ROOT}/mask_images/{self.exp_dir}"
        self.INSTANCE_DIR = f"{ROOT_OF_MASKINGS}/{full_img_name}/{self.mode}_{self.mask_rate}"
        os.makedirs(self.INSTANCE_DIR, exist_ok=True)
        
        if not os.path.exists(f"{self.INSTANCE_DIR}/masking_results.xlsx"):
            print("PHASE 1 -> PATCHES MAPPING")
            instances = [i for i in os.listdir(f"{XAI_ROOT}/explanations/patches_{self.block_width}x{self.block_height}_removal/{self.exp_dir}") if full_img_name in i]
            
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(self.find_patches_coordinates, instances)
            
            all_results = [item for sublist in results for item in sublist]
            df = pd.DataFrame(all_results, columns=["Instance_Block", "Score", "Coordinates"])
            df = df.sort_values(by="Score", ascending=False)
            df.to_csv(f"{self.INSTANCE_DIR}/masking_results.xlsx", index=False, header=True)
        
        else:
            print("PHASE 1 SKIPPED -> PATCHES MAPPING ALREADY AVAILABLE")
            df = pd.read_csv(f"{self.INSTANCE_DIR}/masking_results.xlsx", header=0)

        print("PHASE 2 -> MASKING PROCESS")
        full_img_area = self.full_img_width * self.full_img_height   
        full_img_to_mask = deepcopy(full_img)
        full_img_to_mask_tensor = T.ToTensor()(full_img_to_mask)
        
        os.makedirs(f"{self.INSTANCE_DIR}/removed_patches", exist_ok=True)
        
        check_array = np.ones(shape=(self.full_img_width, self.full_img_height))
        masked_patches, masked_area, idx  = 0, 0, 0
        
        end_condition = False
        while not end_condition:
            left, top, right, bottom = None, None, None, None
            match self.mode:
                case "saliency":
                    left, top, right, bottom = df.iloc[idx]["Coordinates"]
                    
                    if (right - left + 1 != self.block_width) or (bottom - top + 1 != self.block_height):
                        print(f"Skipped patch {idx} due to wrong dimensions")
                        idx += 1
                        continue
                
                case "random":
                    left = np.random.randint(0, self.full_img_width - self.block_width)
                    right = left + self.block_width
                    top = np.random.randint(0, self.full_img_height - self.block_height)
                    bottom = top + self.block_height
            
            if (self.mode == "saliency") and (df.iloc[idx]["Score"] <= 0):
                end_condition = True
            
            else:
                for channel, mean_v in enumerate(self.training_mean):
                    full_img_to_mask_tensor[channel, top:bottom, left:right] = mean_v
                
                check_array[top:bottom, left:right] = 0
                idx, masked_patches = idx + 1, masked_patches + 1
                
                patch = full_img.crop((left, top, right, bottom))
                patch.save(f"{self.INSTANCE_DIR}/removed_patches/patch_{masked_patches}.jpg")
                
                masked_area = np.count_nonzero(check_array == 0)
                if (masked_area > full_img_area * self.mask_rate):
                    end_condition = True
                    
        full_img_masked_tensor_copy = deepcopy(full_img_to_mask_tensor)
        full_img_masked = T.ToPILImage()(full_img_masked_tensor_copy)
        full_img_masked.save(f"{self.INSTANCE_DIR}/{full_img_name}_masked_{self.mode}_{self.mask_rate}{full_img_type}")
        
        print(f"End of Masking Process for the current Instance -> Masked Patches: {masked_patches}\n")
        
        masked_area_ratio = masked_area / full_img_area
        return masked_area_ratio

    def find_patches_coordinates(self, i) -> list:
        filename_parts = i.split("_")
        h_cut, v_cut = int(filename_parts[1]), int(filename_parts[2])
        
        left_b = v_cut * (902 - self.h_overlap)
        top_b = h_cut * (1279 - self.v_overlap)

        mask = Image.open(f"{XAI_ROOT}/def_mask_{self.block_width}x{self.block_height}.png")
        mask_array = np.array(mask)
        
        with open(f"{XAI_ROOT}/explanations/patches_{self.block_width}x{self.block_height}_removal/{self.exp_dir}/{i}/{i}_scores.pkl", "rb") as f: 
            base_scores = pickle.load(f)
        reduced_scores = reduce_scores(mask, base_scores)

        results = list()

        for idx, score in reduced_scores.items():
            if score != [np.nan]:
                instance_patch = i + "_block" + str(idx)
                left_b_patch, top_b_patch, right_b_patch, bottom_b_patch = np.inf, np.inf, -np.inf, -np.inf

                for r in range(mask_array.shape[0]):
                    for c in range(mask_array.shape[1]):
                        if mask_array[r][c] == idx:
                            left_b_patch = min(left_b_patch, c)
                            top_b_patch = min(top_b_patch, r)
                            right_b_patch = max(right_b_patch, c)
                            bottom_b_patch = max(bottom_b_patch, r)
                
                left_b_patch = left_b + left_b_patch
                top_b_patch = top_b + top_b_patch
                right_b_patch = left_b + right_b_patch
                bottom_b_patch = top_b + bottom_b_patch

                results.append([instance_patch, score, (left_b_patch, top_b_patch, right_b_patch, bottom_b_patch)])
        
        print(f"Processed Instance: {i}")
        return results
    
    def __call__(self):
        print(f"*** BEGINNING OF MASKING PROCESS FOR TEST: {self.test_id} ***")
        print(f"*** MODE = {self.mode}, RATE = {self.mask_rate} ***")
        
        DATASET = self.exp_metadata["DATASET"]
        TEST_ID = self.exp_metadata["TEST_ID"]
        MODEL_TYPE = self.exp_metadata["MODEL_TYPE"]
        EXP_METADATA_PATH = f"{LOG_ROOT}/{TEST_ID}-metadata.json"
        
        masking_metadata = dict()
        masking_metadata["FULL_INSTANCES"] = dict()
        
        self.exp_metadata[f"MASK_PROCESS_{self.mode}_{self.mask_rate}_{self.xai_algorithm}_METADATA"] = masking_metadata
        with open(EXP_METADATA_PATH, "w") as jf: json.dump(self.exp_metadata, jf, indent=4)
        
        for inst, path in zip(self.instances, self.paths): 
            masked_area_ratio = self.mask_full_instance(inst, path)
            
            inst_name, c, inst_type = None, None, inst[-4:]
            if DATASET == "CEDAR_Letter":
                # Ex: 0001a.jpg
                inst_name = inst[:-4]               # -> 0001
                c = int(inst_name[:-1])             # -> 1
            if DATASET == "CVL":
                # Ex: 0001-1.png
                inst_name = inst[:-4]               # -> 0001-1 
                c = int(inst_name[:-2])             # -> 1
            if DATASET == "VatLat653":
                # Ex: 0001a-0004r.png
                inst_name = inst[:-4]               # -> 0001a-0004r
                c = int(inst_name[:-7])             # -> 1
                        
            src_path = f"{self.INSTANCE_DIR}/{inst_name}_masked_{self.mode}_{self.mask_rate}{inst_type}"
            dest_dir = f"{DATASET_ROOT}/{DATASET}/{c}-{TEST_ID}_{MODEL_TYPE}_masked_{self.mode}_{self.mask_rate}_{self.xai_algorithm}"
            os.makedirs(dest_dir, exist_ok=True)
            os.system(f"mv {src_path} {dest_dir}/{inst_name}{inst_type}")
            
            self.exp_metadata[f"MASK_PROCESS_{self.mode}_{self.mask_rate}_{self.xai_algorithm}_METADATA"]["FULL_INSTANCES"][inst] = masked_area_ratio
            with open(EXP_METADATA_PATH, "w") as jf: json.dump(self.exp_metadata, jf, indent=4)
            
        print(f"*** END OF MASKING PROCESS FOR TEST: {self.test_id} ***\n")
        
        self.exp_metadata[f"MASK_PROCESS_{self.mode}_{self.mask_rate}_{self.xai_algorithm}_METADATA"]["END_TIMESTAMP"] = str(datetime.now())
        with open(EXP_METADATA_PATH, "w") as jf: json.dump(self.exp_metadata, jf, indent=4)