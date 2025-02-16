import pickle, os, torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from captum.attr import Lime
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnRidge
from captum.attr import visualization as viz
from typing import Tuple

from xai.utils.explanations_utils import reduce_scores, assign_attr_scores_to_mask, custom_visualization, get_rois, return_erased_crops
from xai.utils.image_utils import create_image_grid, apply_transforms_crops, load_rgb_mean_std

XAI_ROOT = "./xai"
torch.backends.cudnn.benchmark = True

### ################## ###
### LimeBase EXPLAINER ###
### ################## ###
class LimeBaseExplainer:
    def __init__(self, classifier: str, test_id: str, dir_name: str,
            block_size: Tuple[int, int], model, surrogate_model: str="LinReg",
            mean=None, std=None, device=None):
        
        self.classifier = classifier
        self.test_id = test_id
        self.dir_name = dir_name
        self.block_width, self.block_height = block_size
        self.model = model
        self.surrogate_model = surrogate_model
        if device is not None: self.device = device
        
        if mean is None and std is None: self.mean_, self.std_ = load_rgb_mean_std(f"./../classifiers/{classifier}/tests/{test_id}")
        else: self.mean_, self.std_ = mean, std

        os.makedirs(f"{XAI_ROOT}/data", exist_ok=True)
    
    def compute_superpixel_scores(self, base_img, base_mask, instance_name, label_idx, n_iter, crop_size, overlap):
        instance_name = instance_name[:-4]
        scores_name = f"{instance_name}_scores.pkl"
        output_dir = f"{XAI_ROOT}/explanations/patches_{self.block_width}x{self.block_height}_removal/{self.dir_name}/{instance_name}"
        os.makedirs(output_dir, exist_ok=True)

        scores = defaultdict(list)

        # Lime Inizialization
        interpretable_model = None
        match self.surrogate_model:
            case "LinReg": interpretable_model = SkLearnLinearRegression()
            case "Ridge": interpretable_model = SkLearnRidge(alpha=1)

        exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)
        lime = Lime(forward_func=self.model, interpretable_model=interpretable_model, similarity_func=exp_eucl_distance)

        if not os.path.exists(f"{output_dir}/{scores_name}"):
            G, nc, nr = create_image_grid(crop_size, overlap, base_img)
            
            # Iterate over the Grid and process each crop
            with tqdm(total=nr*nc, desc="Crop Processing") as pbar:
                for i in range(nr):
                    for j in range(nc):
                        pbar.update(1)
                        x0, y0, x1, y1 = G[f"{i}_{j}"]
                        img_crop = base_img.crop((x0, y0, x1, y1))
                        mask_crop = base_mask.crop((x0, y0, x1, y1))
                        mask_array = np.array(mask_crop)
                        idxs = np.unique(mask_array)

                        img, mask, _ = apply_transforms_crops(img_crop, mask_crop, self.mean_, self.std_)
                        img, mask = img.to(self.device), mask.to(self.device)

                        input_ = img.unsqueeze(0)
                        feature_mask = mask.unsqueeze(0)

                        attr_map_mean = np.zeros((crop_size, crop_size, 3))

                        for _ in range(n_iter):
                            attrs = lime.attribute(input_, target=label_idx,
                                feature_mask=feature_mask, n_samples=40,
                                perturbations_per_eval=16, show_progress=False).squeeze(0)

                            attr_map = attrs.permute(1, 2, 0).cpu().numpy()
                            attr_map_mean += attr_map
                    
                        attr_map_mean /= n_iter

                        norm_attr = viz._normalize_image_attr(attr_map_mean, "all")
                        norm_attr_3d = np.repeat(norm_attr[:, :, None], 3, axis=2)

                        for idx in idxs:
                            super_pixel = norm_attr_3d[mask_array == idx]
                            if len(super_pixel) == self.block_width*self.block_height:
                                scores[idx].append(np.mean(super_pixel))
            
            with open(f"{output_dir}/{scores_name}", "wb") as handle:
                pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def visualize_superpixel_scores_outcomes(self, base_img, base_mask, instance_name, reduction_method, min_eval):  
        instance_name = instance_name[:-4]
        scores_name = f"{instance_name}_scores.pkl"
        output_dir = f"{XAI_ROOT}/explanations/patches_{self.block_width}x{self.block_height}_removal/{self.dir_name}/{instance_name}"

        with open(f"{output_dir}/{scores_name}", "rb") as handle:
            scores = pickle.load(handle)
    
        simplified_scores = reduce_scores(base_mask, scores, reduction_method, min_eval)
        mask_with_attr_scores = assign_attr_scores_to_mask(base_mask, simplified_scores)
        custom_visualization(mask_with_attr_scores, min_eval, f"{output_dir}/{instance_name}")

        try:
            get_rois(mask_with_attr_scores, base_img, base_mask, self.block_width, self.block_height, instance_name, output_dir, num_rois = 5, threshold = 0.5)
        except:
            print("Error in ROIs visualization")

    def compute_masked_patches_explanation(self, base_img, base_mask, instance_name, label_idx, crops_bbxs, reduction_method, min_eval, num_samples_for_baseline=10, save_crops=False):
        instance_name = instance_name[:-4]
        scores_name = f"{instance_name}_scores.pkl"
        output_dir = f"{XAI_ROOT}/explanations/patches_{self.block_width}x{self.block_height}_removal/{self.dir_name}/{instance_name}"

        img_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean_, self.std_)
        ])

        with open(f"{output_dir}/{scores_name}", "rb") as handle:
            scores = pickle.load(handle)

        simplified_scores = reduce_scores(base_mask, scores, reduction_method, min_eval)

        dict_plots_template = {
            'num_patches_to_remove': [],
            'clean_crop': [],
            'relevant_patches': [],
            'misleading_patches': [],
            'random_patches': [],
            'random_patches_std': [],
        }

        for i, bbx in enumerate(crops_bbxs):
            os.makedirs(f"{output_dir}/{instance_name}_crop{str(i)}", exist_ok=True)

            dict_plots = deepcopy(dict_plots_template)
            img_crop, mask_crop = base_img.crop(bbx), base_mask.crop(bbx)
            mask_with_attr_scores = assign_attr_scores_to_mask(mask_crop, simplified_scores)
            custom_visualization(mask_with_attr_scores, min_eval, f"{output_dir}/{instance_name}_crop{str(i)}/{instance_name}_crop{str(i)}")

            mask_crop_array = np.array(mask_crop)
            idxs = np.unique(mask_crop_array)
            valid_idxs = [idx for idx in idxs if np.sum(mask_crop_array == idx) == self.block_width * self.block_height]

            filtered_simplified_scores = {k: v for k, v in simplified_scores.items() if k in valid_idxs}
            sorted_filtered_simplified_scores = dict(sorted(filtered_simplified_scores.items(), key=lambda item: item[1], reverse=True))

            n_patches_to_remove = 1
            while True:
                try:
                    erased_crops = return_erased_crops(n_patches_to_remove, num_samples_for_baseline, sorted_filtered_simplified_scores, mask_crop_array, img_crop)
                    erased_crops.insert(0, img_crop)

                    list_results = list()

                    img_crop.save(f"{output_dir}/{instance_name}_crop{str(i)}/{instance_name}_crop{str(i)}.png")

                    for k, erased_crop in enumerate(erased_crops):
                        if save_crops:
                            erased_crop.save(f"{output_dir}/{instance_name}_crop{str(i)}/{instance_name}_crop{str(i)}_{str(k+1)}patches.png")
                    
                        input_ = img_transforms(erased_crop).to(self.device).unsqueeze(0)

                        with torch.no_grad():
                            if self.classifier == "classifier_ResNet18":
                                output = self.model(input_)
                                output_probs = F.softmax(output, dim=1)[0]

                            class_confidence = output_probs[label_idx].detach().cpu().item()
                            predicted_class = output_probs.argmax().detach().cpu().item()
                            predicted_confidence = output_probs.max().detach().cpu().item()
                            list_results.append([label_idx, class_confidence, predicted_class, predicted_confidence])
               
                    clean_crop_confidence_gt, clean_crop_confidence_pred = round(list_results[0][1]*100, 2), round(list_results[0][3]*100, 2)
                    relevant_patches_confidence_gt, relevant_patches_confidence_pred = round(list_results[1][1]*100, 2), round(list_results[1][3]*100, 2) 
                    misleading_patches_confidence_gt, misleading_patches_confidence_pred = round(list_results[2][1]*100, 2), round(list_results[2][3]*100, 2)

                    random_confidences_gt = [result[1] for result in list_results[3:]]
                    random_confidences_pred = [result[3] for result in list_results[3:]]
                    random_samples_counter = len(random_confidences_gt)
                    mean_confidence_gt = np.mean(random_confidences_gt) if random_samples_counter > 0 else 0.0
                    confidence_std_gt = np.std(random_confidences_gt) if random_samples_counter > 0 else 0.0

                    mean_confidence_pred = np.mean(random_confidences_pred) if random_samples_counter > 0 else 0.0
                    confidence_std_pred = np.std(random_confidences_pred) if random_samples_counter > 0 else 0.0

                    mean_confidence_gt, mean_confidence_pred = round(mean_confidence_gt * 100, 2), round(mean_confidence_pred * 100, 2)
                    confidence_std_gt, confidence_std_pred = round(confidence_std_gt * 100, 2), round(confidence_std_pred * 100, 2)

                    with open(f"{output_dir}/{instance_name}_crop{str(i)}/crop{str(i)}_confidence_patches.txt", "a") as f:
                        f.write("CLEAN CROP\n")
                        f.write(f"True class {list_results[0][0]} - Confidence: {clean_crop_confidence_gt}%\n")
                        f.write(f"Predicted class {list_results[0][2]} - Confidence: {clean_crop_confidence_pred}%\n")
                        f.write(f"REMOVING THE {n_patches_to_remove} MOST RELEVANT PATCHES\n")
                        f.write(f"True class {list_results[1][0]} - Confidence: {relevant_patches_confidence_gt}%\n")
                        f.write(f"Predicted class {list_results[1][2]} - Confidence: {relevant_patches_confidence_pred}%\n")
                        f.write(f"REMOVING THE {n_patches_to_remove} MOST MISLEADING PATCHES\n")
                        f.write(f"True class {list_results[2][0]} - Confidence: {misleading_patches_confidence_gt}%\n")
                        f.write(f"Predicted class {list_results[2][2]} - Confidence: {misleading_patches_confidence_pred}%\n")
                        f.write(f"REMOVING {n_patches_to_remove} RANDOM PATCHES ({random_samples_counter}/{num_samples_for_baseline} repetitions)\n")
                        f.write(f"True class {list_results[3][0]} - Mean Confidence: {mean_confidence_gt}% - Confidence Standard Deviation: {confidence_std_gt}%\n")
                        f.write(f"Predicted class {list_results[3][2]} - Mean Confidence: {mean_confidence_pred}% - Confidence Standard Deviation: {confidence_std_pred}%\n")
                        f.write('---------------------\n')
                    
                    dict_plots['clean_crop'].append(clean_crop_confidence_gt)
                    dict_plots['relevant_patches'].append(relevant_patches_confidence_gt)
                    dict_plots['misleading_patches'].append(misleading_patches_confidence_gt)
                    dict_plots['random_patches'].append(mean_confidence_gt)
                    dict_plots['random_patches_std'].append(confidence_std_gt)
                
                except:
                    print(f"End of patch removing for crop {i} with {n_patches_to_remove-1} removals")
                    break
                
                dict_plots['num_patches_to_remove'].append(n_patches_to_remove)
                n_patches_to_remove += 1
                
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlabel('Number of patches removed [-]')
            ax.set_ylabel('Confidence [%]')
            ax.plot(dict_plots['num_patches_to_remove'], dict_plots['clean_crop'], label='Clean crop confidence')
            ax.plot(dict_plots['num_patches_to_remove'], dict_plots['relevant_patches'], marker='o', label='Relevant patches')
            ax.plot(dict_plots['num_patches_to_remove'], dict_plots['misleading_patches'], marker='o', label='Misleading patches')
            ax.errorbar(dict_plots['num_patches_to_remove'], dict_plots['random_patches'], dict_plots['random_patches_std'], marker='o', label='Random patches')
        
            ax.legend(loc='lower left')
            fig.savefig(f'{output_dir}/{instance_name}_crop{str(i)}/{instance_name}_crop{i}_plot.png')
            plt.close(fig)