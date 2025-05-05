import os, json
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from utils import get_train_instance_patterns, get_test_instance_patterns

from xai.explainers.lime_base_explainer import LimeBaseExplainer
from xai.explainers.glime_binomial_explainer import GLimeBinomialExplainer
from xai.utils.image_utils import create_image_grid

XAI_ROOT = "./xai"

### ############################## ###
### EXPLANATIONS SUPPORT FUNCTIONS ###
### ############################## ###
def get_instances_to_explain(dataset, source, class_to_idx, phase):
    instances, labels = list(), list()
    
    pattern = None
    if phase == "train": pattern = get_train_instance_patterns()
    elif phase == "test": pattern = get_test_instance_patterns()
    
    for f in os.listdir(source):
        if pattern[dataset](f):
            writer_id = int(f[:4])
            label = class_to_idx[str(writer_id)]
        
            instances.append(f"{source}/{f}")
            labels.append(label)

    return instances, labels

def normalize_scores(scores, percentile=99):
    keys, values = list(scores.keys()), np.array(list(scores.values()))
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=False)}
    sorted_values = np.array(list(sorted_scores.values()))
    
    cum_sums = np.cumsum(np.abs(sorted_values))
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    threshold = sorted_values[threshold_id]
    
    norm_values = np.clip(values / threshold, -1, 1)
    norm_scores = {k: v for k, v in zip(keys, norm_values)}
    
    return norm_scores

def reduce_scores(scores, reduction_method="mean"):
    reductions = {"mean": np.mean, "median": np.median}
    red_func = reductions.get(reduction_method, np.mean)

    reduced_scores = dict()
    for k in scores.keys(): reduced_scores[str(k)] = float(red_func(scores[k]))
    
    return reduced_scores
    
def assign_attr_scores_to_mask(base_mask, scores):
    base_mask_array = np.array(deepcopy(base_mask)).astype(np.float32)

    for key in list(scores.keys()):
        base_mask_array[base_mask_array == float(key)] = scores[key]

    return base_mask_array

def setup_explainer(xai_algorithm, surrogate_model, model_type, model, num_samples, batch_size, kernel_width, mean, std):
    if xai_algorithm == "LimeBase":
        return LimeBaseExplainer(model_type, model, surrogate_model, mean, std, num_samples, batch_size, kernel_width)
    elif xai_algorithm == "GLimeBinomial":
        return GLimeBinomialExplainer(model_type, model, surrogate_model, mean, std, num_samples, batch_size, kernel_width)

def explain_instance(explainer, img, img_rows, img_cols, instance_name, label, mask, mask_rows, mask_cols, output_dir, crop_size, overlap, iters):
    scores = dict()
    total_masked_patches = list()
    total_samples = dict()
    for i in range(0, (mask_rows)*(mask_cols)): scores[i] = list()
    
    grid_dict = create_image_grid(img_rows, img_cols, crop_size, overlap)

    with tqdm(total=img_rows*img_cols, desc="Crop Processing") as pbar:
        for r in range(0, img_rows):
            for c in range(0, img_cols):
                coordinates = grid_dict[f"{r}_{c}"]
                crop_img, crop_mask = img.crop(coordinates), mask.crop(coordinates)
                crop_mask_array = np.array(crop_mask)
                segments = np.array(crop_mask_array/100, dtype=np.uint16)
                
                crop_attr = dict()

                for iteration in range(0, iters):
                    attr, maskings, binary_samples = explainer.explain_instance(crop_img, segments, label)
                    for k, v in attr.items():
                        if k not in crop_attr: crop_attr[k] = v
                        else: crop_attr[k] += v
                
                for k, v in crop_attr.items(): crop_attr[k] = float(v / iters)
                
                normalized_crop_attr = normalize_scores(crop_attr, percentile=98)
                for k, v in normalized_crop_attr.items(): scores[k].append(v)

                total_masked_patches.extend(maskings)
                for i, s in enumerate(binary_samples): 
                    total_samples[f"{instance_name}_gridcrop{r}_{c}_num{i}"] = s

                pbar.update(1)

    final_scores = reduce_scores(scores)
    with open(f"{output_dir}/{instance_name}_scores.json", "w") as f: json.dump(final_scores, f, indent=4)

    return final_scores, np.array(total_masked_patches), total_samples

def visualize_exp_outcome(scores, mask, instance_name, output_dir,):
    mask_array = np.array(mask)/100
    for k in scores.keys(): mask_array[mask_array == float(k)] = scores[k]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])
    cmap.set_bad(color='black')

    vmin, vmax = -1, 1
    heat_map = ax.imshow(mask_array, cmap=cmap, vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    fig.colorbar(heat_map, orientation="horizontal", cax=cax)

    output_path = f"{output_dir}/{instance_name}_attr_map.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)