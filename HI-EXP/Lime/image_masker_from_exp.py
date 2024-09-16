import pickle, os
import numpy as np
import pandas as pd
import multiprocessing as mp
from PIL import Image
from copy import deepcopy
from masked_patches_explainer import reduce_scores
from argparse import ArgumentParser
from tqdm import tqdm
from torchvision import transforms as T

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def get_args():
    parser = ArgumentParser()
    parser.add_argument("-instance", type=str)
    parser.add_argument("-test_id", type=str)
    parser.add_argument("-model", type=str)
    parser.add_argument("-timestamp", type=str)
    parser.add_argument("-surrogate_model", type=str)

    return parser.parse_args()

### ############### ###
### PRINCIPAL CLASS ###
### ############### ###

class ImageMaskerFromExp:
    def __init__(self, instance: str, test_id: str, model: str, timestamp: str, surrogate_model: str):
        self.instance = instance
        self.test_id = test_id
        self.model = model
        self.timestamp = timestamp
        self.surrogate_model = surrogate_model
    
    def setup_files_for_masking(self):
        os.makedirs(f"./mask_images/{self.test_id}_{self.model}_{self.timestamp}", exist_ok=True)

        self.INSTANCE_TO_MASK_DIRECTORY = f"./mask_images/{self.test_id}_{self.model}_{self.timestamp}/{self.instance}"
        os.makedirs(self.INSTANCE_TO_MASK_DIRECTORY, exist_ok=True)
        os.makedirs(f"{self.INSTANCE_TO_MASK_DIRECTORY}/instances", exist_ok=True)

        self.INSTANCE_FILETYPE = None
        if f"{self.instance}.jpg" in os.listdir("./mask_images"): self.INSTANCE_FILETYPE = ".jpg"
        elif f"{self.instance}.png" in os.listdir("./mask_images"): self.INSTANCE_FILETYPE = ".png"
        else: raise Exception("Instance not found in 'mask_images' directory")
        
        os.system(f"cp ./mask_images/{self.instance}{self.INSTANCE_FILETYPE} {self.INSTANCE_TO_MASK_DIRECTORY}/{self.instance}{self.INSTANCE_FILETYPE}")

        EXP_DIR = f"./explanations/patches_75x75_removal/{self.test_id}-{self.model}-{self.timestamp}-{self.surrogate_model}"
        
        all_instances = os.listdir(EXP_DIR)
        filtered_instances = [i for i in all_instances if self.instance in i]

        for i in tqdm(filtered_instances, desc="Setup of Files for Masking Process"):
            os.makedirs(f"{self.INSTANCE_TO_MASK_DIRECTORY}/instances/{i}", exist_ok=True)
            os.system(f"cp {EXP_DIR}/{i}/{i}_scores.pkl {self.INSTANCE_TO_MASK_DIRECTORY}/instances/{i}/{i}_scores.pkl")
            os.system(f"cp ./explanations/page_level/{i}/{i}_mask_blocks_75x75.png {self.INSTANCE_TO_MASK_DIRECTORY}/instances/{i}/{i}_mask_blocks_75x75.png")
    
    def mask_full_instance(self, ratios=[0.01, 0.05, 0.1]):
        print(f"\nBEGINNING OF THE MASKING PROCESS FOR INSTANCE: {self.instance}")

        full_img = Image.open(f"{self.INSTANCE_TO_MASK_DIRECTORY}/{self.instance}{self.INSTANCE_FILETYPE}")
        full_img_width, full_img_height = full_img.size
        self.INSTANCE_WIDTH, self.INSTANCE_HEIGHT = 902, 1279

        vertical_cuts = (full_img_width // self.INSTANCE_WIDTH)*3
        horizontal_cuts = (full_img_height // self.INSTANCE_HEIGHT)*3

        self.H_OVERLAP = int((((vertical_cuts+1)*self.INSTANCE_WIDTH) - full_img_width) / vertical_cuts)
        self.V_OVERLAP = int((((horizontal_cuts+1)*self.INSTANCE_HEIGHT) - full_img_height) / horizontal_cuts)
        
        if not os.path.exists(f"{self.INSTANCE_TO_MASK_DIRECTORY}/{self.instance}_masking_results.xlsx"):
            print("PHASE 1 -> PATCHES MAPPING")
            instances = os.listdir(f"{self.INSTANCE_TO_MASK_DIRECTORY}/instances")

            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(self.find_patches_coordinates, instances)

            all_results = [item for sublist in results for item in sublist]
            df = pd.DataFrame(all_results, columns=["Instance_Block", "Score", "Coordinates"])
            df.to_excel(f"{self.INSTANCE_TO_MASK_DIRECTORY}/{self.instance}_masking_results.xlsx", index=False, header=False)
        
        else:
            print("PHASE 1 SKIPPED -> PATCHES MAPPING ALREADY AVAILABLE")
            df = pd.read_excel(f"{self.INSTANCE_TO_MASK_DIRECTORY}/{self.instance}_masking_results.xlsx")

        print("PHASE 2 -> MASKING PROCESS")
        df_sorted = df.sort_values(by="Score", ascending=False)
        full_img_area = full_img_width * full_img_height
        patch_area = 75*75

        patches_to_mask = dict()
        for ratio in ratios:
            num_patches_to_mask = int((full_img_area * ratio) / patch_area)
            patches_to_mask[ratio] = num_patches_to_mask
        
        masked_patches, idx, end_condition = 0, 0, False
        full_img_to_mask = deepcopy(full_img)
        full_img_to_mask_tensor = T.ToTensor()(full_img_to_mask)
        os.makedirs(f"{self.INSTANCE_TO_MASK_DIRECTORY}/removed_patches", exist_ok=True)
        
        while not end_condition:
            left, top, right, bottom = df_sorted.iloc[idx]["Coordinates"]
            
            if (right - left + 1 != 75) or (bottom - top + 1 != 75):
                print(f"Skipped patch {idx} due to wrong dimensions")
                idx += 1
                continue
            
            full_img_to_mask_tensor[:, top:bottom, left:right] = 0
            idx, masked_patches = idx + 1, masked_patches + 1
            full_img.crop((left, top, right, bottom)).save(f"{self.INSTANCE_TO_MASK_DIRECTORY}/removed_patches/{self.instance}_removed_patch_{masked_patches}{self.INSTANCE_FILETYPE}")
            
            if masked_patches == patches_to_mask[ratios[0]]:
                full_img_masked_tensor_copy = deepcopy(full_img_to_mask_tensor)
                full_img_masked = T.ToPILImage()(full_img_masked_tensor_copy)
                full_img_masked.save(f"{self.INSTANCE_TO_MASK_DIRECTORY}/{self.instance}_masked_{ratios[0]}{self.INSTANCE_FILETYPE}")
                print(f"End of Masking Process for Ratio {ratios[0]} -> Masked Patches: {masked_patches}")
            elif masked_patches == patches_to_mask[ratios[1]]:
                full_img_masked_tensor_copy = deepcopy(full_img_to_mask_tensor)
                full_img_masked = T.ToPILImage()(full_img_masked_tensor_copy)
                full_img_masked.save(f"{self.INSTANCE_TO_MASK_DIRECTORY}/{self.instance}_masked_{ratios[1]}{self.INSTANCE_FILETYPE}")
                print(f"End of Masking Process for Ratio {ratios[1]} -> Masked Patches: {masked_patches}")
            elif masked_patches == patches_to_mask[ratios[2]]:
                full_img_masked_tensor_copy = deepcopy(full_img_to_mask_tensor)
                full_img_masked = T.ToPILImage()(full_img_masked_tensor_copy)
                full_img_masked.save(f"{self.INSTANCE_TO_MASK_DIRECTORY}/{self.instance}_masked_{ratios[2]}{self.INSTANCE_FILETYPE}")
                print(f"End of Masking Process for Ratio {ratios[2]} -> Masked Patches: {masked_patches}")
                end_condition = True
        
        print(f"END OF THE MASKING PROCESS FOR INSTANCE: {self.instance}")

    def find_patches_coordinates(self, i):
        filename_parts = i.split("_")
        h_cut, v_cut = int(filename_parts[1]), int(filename_parts[2])

        left_b = (v_cut * self.INSTANCE_WIDTH) - (v_cut * self.H_OVERLAP)
        top_b = (h_cut * self.INSTANCE_HEIGHT) - (h_cut * self.V_OVERLAP)

        mask = Image.open(f"{self.INSTANCE_TO_MASK_DIRECTORY}/instances/{i}/{i}_mask_blocks_75x75.png")
        mask_array = np.array(mask)
        base_scores = pickle.load(open(f"{self.INSTANCE_TO_MASK_DIRECTORY}/instances/{i}/{i}_scores.pkl", "rb"))
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

### #### ###
### MAIN ###
### #### ###
if __name__ == '__main__':
    args = get_args()
    masker = ImageMaskerFromExp(
        instance=args.instance,
        test_id=args.test_id,
        model=args.model,
        timestamp=args.timestamp,
        surrogate_model=args.surrogate_model)
    
    masker.setup_files_for_masking()
    masker.mask_full_instance()