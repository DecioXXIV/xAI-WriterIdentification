import pickle, cv2, os, random, torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from captum.attr import Lime
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import visualization as viz
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from typing import Tuple
from explanations_utils import create_image_grid
from image_utils import apply_transforms_crops, load_rgb_mean_std

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def reduce_scores(base_mask, scores, reduction_method = 'mean', min_eval = 10):

    simplified_scores = deepcopy(scores)

    base_mask_array = np.array(deepcopy(base_mask))
    idxs = np.unique(base_mask_array)

    for idx in idxs:
        
        if (idx not in list(simplified_scores.keys())) or (len(simplified_scores[idx]) < min_eval):
            simplified_scores[idx] = [np.nan]
        
        else:
            if reduction_method == 'mean':
                simplified_scores[idx] = np.mean(simplified_scores[idx])
            elif reduction_method == 'median':
                simplified_scores[idx] = np.median(simplified_scores[idx])

    return simplified_scores

def assign_attr_scores_to_mask(base_mask, scores):

    base_mask_array = np.array(deepcopy(base_mask)).astype(np.float32)

    for key in list(scores.keys()):
        base_mask_array[base_mask_array == float(key)] = scores[key]

    return base_mask_array

def custom_visualization(
    norm_attr,
    min_eval,
    output_name = '',
    fig_size: Tuple[int, int] = (6, 6)
    ):
    
    plt_fig = Figure(figsize=fig_size)
    plt_axis = plt_fig.subplots()
    
    plt_axis.xaxis.set_ticks_position("none")
    plt_axis.yaxis.set_ticks_position("none")
    plt_axis.set_yticklabels([])
    plt_axis.set_xticklabels([])
    plt_axis.grid(visible=False)

    cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])
    cmap.set_bad(color='black')
    vmin, vmax = -1, 1
    heat_map = plt_axis.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax)
    
    axis_separator = make_axes_locatable(plt_axis)
    colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
    plt_fig.colorbar(heat_map, orientation="horizontal", cax=colorbar_axis)

    plt_fig.savefig(f'{output_name}_att_heat_map_{min_eval}.png')

def get_rois(scores_matrix, page, mask, block_width, block_height, pagename, output_dir, num_rois: int = None, threshold = 0.5):

    if not num_rois == None:
        flat_matrix = scores_matrix.flatten()
        flat_matrix_no_nan = np.unique(flat_matrix[np.logical_not(np.isnan(flat_matrix))])
        threshold = np.sort(flat_matrix_no_nan)[-num_rois]

    logical_matrix = np.greater_equal(scores_matrix, np.ones_like(scores_matrix)*threshold)
    logical_matrix = logical_matrix.astype(np.uint8)

    cnts = cv2.findContours(logical_matrix*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    
    img = np.array(deepcopy(page))
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_rgb_backup = deepcopy(im_rgb)    
    
    for z, c in enumerate(cnts):
        cv2.drawContours(im_rgb, c, -1, (0,255,0), 2)
        # bottomLeftCornerOfText = (np.max(c[:,:,0]), np.min(c[:,:,1]))
        # cv2.putText(im_rgb, str(z), bottomLeftCornerOfText, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 2)
    
    cv2.imwrite(f'{output_dir}/{pagename}_rois_t_{str(threshold)}.png', im_rgb)
    
    base_mask_array = np.array(deepcopy(mask)) + np.ones_like(scores_matrix)
    roi_matrix = np.multiply(logical_matrix, base_mask_array)

    diff_shapes = list(np.array(roi_matrix.shape) - np.array(im_rgb_backup.shape[:2]))
    
    if diff_shapes[0] > 0:
        roi_matrix = roi_matrix[:-diff_shapes[0],:]
    elif diff_shapes[0] < 0:
       im_rgb_backup = im_rgb_backup[:diff_shapes[0],:] 

    if diff_shapes[1] > 0:
        roi_matrix = roi_matrix[:,:-diff_shapes[1]]
    elif diff_shapes[1] < 0:
       im_rgb_backup = im_rgb_backup[:,:diff_shapes[1]] 

    roi_idxs = list(np.unique(roi_matrix))
    roi_idxs.remove(0)
    
    for k, idx in enumerate(roi_idxs):
        crop = im_rgb_backup[roi_matrix == idx].reshape(block_width, block_height, 3)
        cv2.imwrite(f'{output_dir}/{pagename}_ROI_{str(k)}.png', crop)

def return_erased_crops(num_patches, num_random_samples, dict_scores, mask_crop_array, crop):
    lists_idxs = [
        list(dict_scores.keys())[:num_patches],
        list(dict_scores.keys())[-num_patches:]
    ]

    for j in range(num_random_samples):
        lists_idxs.append(random.sample(list(dict_scores.keys()), num_patches))

    list_erased_crops = []

    for list_idxs in lists_idxs:
        rois_to_mask = np.zeros_like(mask_crop_array)
        for roi_idx in list_idxs:
            super_pixel = mask_crop_array == roi_idx
            rois_to_mask += super_pixel
        
        rois_to_mask = np.ones_like(rois_to_mask) - rois_to_mask
        rois_to_mask_3d = rois_to_mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]

        crop_array = np.array(deepcopy(crop))
        masked_crop_array = crop_array*rois_to_mask_3d
        masked_crop_pil = Image.fromarray(np.uint8(masked_crop_array))

        list_erased_crops.append(masked_crop_pil)

    return list_erased_crops

def get_crops_bbxs(image, final_width=380, final_height=380):
    img_width, img_height = image.size

    vertical_cuts = img_width // final_width
    horizontal_cuts = img_height // final_height
    
    h_overlap = int((((vertical_cuts+1)*final_width) - img_width) / vertical_cuts)
    v_overlap = int((((horizontal_cuts+1)*final_height) - img_height) / horizontal_cuts)

    crops_bbxs = list()

    for h_cut in range(0, horizontal_cuts+1):
        for v_cut in range(0, vertical_cuts+1):
            left = v_cut*final_width - v_cut*h_overlap
            right = left + final_width
            top = h_cut*final_height - h_cut*v_overlap
            bottom = top + final_height

            crops_bbxs.append((left, top, right, bottom))

    return crops_bbxs

### ######################## ###
### MASKED PATCHES EXPLAINER ###
### ######################## ###
class MaskedPatchesExplainer:
    def __init__(self, 
            classifier: str, 
            test_id: str, 
            block_size: Tuple[int, int], 
            model, 
            mean: Tuple[float,float,float]=None,
            std: Tuple[float,float,float]=None,
            device=None):
        
        self.classifier = classifier
        self.test_id = test_id
        self.block_width, self.block_height = block_size
        self.model = model
        if device is not None:
            self.device = device
        
        if mean is None and std is None:
            self.mean_, self.std_ = load_rgb_mean_std(f"./../{classifier}/tests/{test_id}/train")
        else:
            self.mean_, self.std_ = mean, std
    
    def compute_superpixel_scores(self, instance_name, label_idx, n_iter, crop_size, overlap):
        scores_name = f"{instance_name}_scores.pkl"
        output_dir = f"./explanations/patches_{self.block_width}x{self.block_height}_removal/{self.test_id}/{instance_name}"
        os.makedirs(output_dir, exist_ok=True)
    
        img_path = f"./data/{instance_name}.jpg"
        mask_path = f"./explanations/page_level/{instance_name}/{instance_name}_mask_blocks_{self.block_width}x{self.block_height}.png"

        base_img, base_mask = Image.open(img_path), Image.open(mask_path)
        scores = defaultdict(list)

        exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)

        if not os.path.exists(f"{output_dir}/{scores_name}"):
            G, nc, nr = create_image_grid(crop_size, overlap, base_img)

            for i in tqdm(range(nr), leave=False):
                for j in tqdm(range(nc), leave=False):
                    x0, y0, x1, y1 = G[f"{i}_{j}"]
                    img_crop = base_img.crop((x0, y0, x1, y1))
                    mask_crop = base_mask.crop((x0, y0, x1, y1))
                    mask_array = np.array(mask_crop)
                    idxs = np.unique(mask_array)

                    img, mask, n_interpret_features = apply_transforms_crops(img_crop, mask_crop, self.mean_, self.std_)
                    img, mask = img.to(self.device), mask.to(self.device)

                    input_ = img.unsqueeze(0)
                    feature_mask = mask.unsqueeze(0)

                    attr_map_mean = np.zeros((crop_size, crop_size, 3))

                    lr_lime = Lime(self.model, interpretable_model=SkLearnLinearRegression(), similarity_func=exp_eucl_distance)

                    for _ in range(n_iter):
                        attrs = lr_lime.attribute(
                            input_,
                            target=label_idx,
                            feature_mask=feature_mask,
                            n_samples=40,
                            perturbations_per_eval=16,
                            show_progress=False
                        ).squeeze(0)

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
    
    def visualize_superpixel_scores_outcomes(self, instance_name, reduction_method, min_eval):
        scores_name = f"{instance_name}_scores.pkl"
        output_dir = f"./explanations/patches_{self.block_width}x{self.block_height}_removal/{self.test_id}/{instance_name}"
        
        img_path = f"./data/{instance_name}.jpg"
        mask_path = f"./explanations/page_level/{instance_name}/{instance_name}_mask_blocks_{self.block_width}x{self.block_height}.png"

        base_img, base_mask = Image.open(img_path), Image.open(mask_path)

        with open(f"{output_dir}/{scores_name}", "rb") as handle:
            scores = pickle.load(handle)
    
        simplified_scores = reduce_scores(base_mask, scores, reduction_method, min_eval)
        mask_with_attr_scores = assign_attr_scores_to_mask(base_mask, simplified_scores)
        custom_visualization(mask_with_attr_scores, min_eval, f"{output_dir}/{instance_name}")

        get_rois(mask_with_attr_scores, base_img, base_mask, self.block_width, self.block_height, instance_name, output_dir, num_rois = 5, threshold = 0.5)

    def compute_masked_patches_explanation(self, instance_name, label_idx, crops_bbxs, reduction_method, min_eval, num_samples_for_baseline=10, save_crops=False):
        scores_name = f"{instance_name}_scores.pkl"
        output_dir = f"./explanations/patches_{self.block_width}x{self.block_height}_removal/{self.test_id}/{instance_name}"
    
        img_path = f"./data/{instance_name}.jpg"
        mask_path = f"./explanations/page_level/{instance_name}/{instance_name}_mask_blocks_{self.block_width}x{self.block_height}.png"

        base_img, base_mask = Image.open(img_path), Image.open(mask_path)

        img_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean_, self.std_)
        ])

        with open(f"{output_dir}/{scores_name}", "rb") as handle:
            scores = pickle.load(handle)

        simplified_scores = reduce_scores(base_mask, scores, reduction_method, min_eval)

        dict_plots_template = {
            'num_patches_to_remove': [5, 7, 10, 15, 20, 30],
            'clean_crop': [],
            'relevant_patches': [],
            'misleading_patches': [],
            'random_patches': [],
            'random_patches_std': [],
        }

        for i, bbx in enumerate(crops_bbxs):
            dict_plots = deepcopy(dict_plots_template)
            img_crop, mask_crop = base_img.crop(bbx), base_mask.crop(bbx)
            mask_with_attr_scores = assign_attr_scores_to_mask(mask_crop, simplified_scores)
            custom_visualization(mask_with_attr_scores, min_eval, f"{output_dir}/{instance_name}_{str(i)}")

            mask_crop_array = np.array(mask_crop)
            idxs = np.unique(mask_crop_array)
            valid_idxs = [idx for idx in idxs if np.sum(mask_crop_array == idx) == self.block_width * self.block_height]

            filtered_simplified_scores = {k: v for k, v in simplified_scores.items() if k in valid_idxs}
            sorted_filtered_simplified_scores = dict(sorted(filtered_simplified_scores.items(), key=lambda item: item[1], reverse=True))

            for n_patches_to_remove in dict_plots["num_patches_to_remove"]:
                erased_crops = return_erased_crops(n_patches_to_remove, num_samples_for_baseline, sorted_filtered_simplified_scores, mask_crop_array, img_crop)
                erased_crops.insert(0, img_crop)

                list_results = list()

                img_crop.save(f"{output_dir}/{instance_name}_{str(i)}_0.png")

                for k, erased_crop in enumerate(erased_crops):
                    if save_crops:
                        erased_crop.save(f"{output_dir}/{instance_name}_{str(i)}_{str(k+1)}.png")
                    
                    input_ = img_transforms(erased_crop).to(self.device).unsqueeze(0)

                    with torch.no_grad():
                        output = self.model(input_)
                        output_probs = F.softmax(output, dim=1)[0]
                        predicted_class = output_probs.argmax().cpu().item()
                        confidence = output_probs.max().cpu().item()

                        if predicted_class != label_idx:
                            confidence = 0.0
                        
                        list_results.append([predicted_class, confidence])
                
                clean_crop_confidence = round(list_results[0][1]*100, 2)
                relevant_patches_confidence = round(list_results[1][1]*100, 2)
                misleading_patches_confidence = round(list_results[2][1]*100, 2)

                random_confidences = [result[1] for result in list_results[3:] if result[0] == list_results[0][0]]
                random_samples_counter = len(random_confidences)
                mean_confidence = np.mean(random_confidences) if random_samples_counter > 0 else 0.0
                confidence_std = np.std(random_confidences) if random_samples_counter > 0 else 0.0

                mean_confidence = round(mean_confidence * 100, 2)
                confidence_std = round(confidence_std * 100, 2)

                with open(f"{output_dir}/{str(i)}_confidence_patches.txt", "a") as f:
                    f.write(f'Clean crop: Predicted class {list_results[0][0]} - Confidence: {clean_crop_confidence}%\n')
                    f.write(f'Crop without the {n_patches_to_remove} most relevant patches: Predicted class {list_results[1][0]} - Confidence: {relevant_patches_confidence}%\n')
                    f.write(f'Crop without the {n_patches_to_remove} most misleading patches: Predicted class {list_results[2][0]} - Confidence: {misleading_patches_confidence}%\n')
                    f.write(f'Crop without {n_patches_to_remove} random patches ({random_samples_counter}/{num_samples_for_baseline} repetitions): Predicted class {list_results[0][0]} - Mean Confidence: {mean_confidence}% - Confidence Standard Deviation: {confidence_std}%\n')
                    f.write('---------------------\n')

                dict_plots['clean_crop'].append(clean_crop_confidence)
                dict_plots['relevant_patches'].append(relevant_patches_confidence)
                dict_plots['misleading_patches'].append(misleading_patches_confidence)
                dict_plots['random_patches'].append(mean_confidence)
                dict_plots['random_patches_std'].append(confidence_std)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlabel('Number of patches removed [-]')
            ax.set_ylabel('Confidence [%]')
            ax.plot(dict_plots['num_patches_to_remove'], dict_plots['clean_crop'], label='Clean crop confidence')
            ax.plot(dict_plots['num_patches_to_remove'], dict_plots['relevant_patches'], marker='o', label='Relevant patches')
            ax.plot(dict_plots['num_patches_to_remove'], dict_plots['misleading_patches'], marker='o', label='Misleading patches')
            ax.errorbar(dict_plots['num_patches_to_remove'], dict_plots['random_patches'], dict_plots['random_patches_std'], marker='o', label='Random patches')
        
            ax.legend(loc='lower left')
            fig.savefig(f'{output_dir}/{instance_name}_plot.png')
            plt.close(fig)