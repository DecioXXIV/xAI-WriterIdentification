import cv2, os, random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from typing import Tuple

from utils import get_train_instance_patterns, get_test_instance_patterns

XAI_ROOT = "./xai"

### ############################## ###
### EXPLANATIONS SUPPORT FUNCTIONS ###
### ############################## ###
def get_instances_to_explain(dataset, source, class_to_idx, phase):
    instances, labels = list(), list()
    
    train_patterns = get_train_instance_patterns()
    test_patterns = get_test_instance_patterns()

    pattern_check = train_patterns[dataset] if phase == "train" else test_patterns[dataset]
    
    for f in os.listdir(source):
        if pattern_check(f): continue

        writer_id = int(f[:4])
        label = class_to_idx[str(writer_id)]
        
        src_path, dest_path = f"{source}/{f}", f"{XAI_ROOT}/data/{f}"
        os.system(f"cp {src_path} {dest_path}")
        
        instances.append(f)
        labels.append(label)
    
    return instances, labels

def reduce_scores(base_mask, scores, reduction_method="mean", min_eval=10):
    base_mask_array = np.array(base_mask)
    idxs = np.unique(base_mask_array)
    
    reductions = {"mean": np.mean, "median": np.median}
    red_func = reductions.get(reduction_method, np.mean)
    
    reduced_scores = dict()
    for idx in idxs:
        values = scores.get(idx, [])
        if len(values) < min_eval: reduced_scores[idx] = [np.nan]
        else: reduced_scores[idx] = red_func(values)
    
    return reduced_scores
    
def assign_attr_scores_to_mask(base_mask, scores):
    base_mask_array = np.array(deepcopy(base_mask)).astype(np.float32)

    for key in list(scores.keys()):
        base_mask_array[base_mask_array == float(key)] = scores[key]

    return base_mask_array

def custom_visualization(
    norm_attr: np.ndarray,
    min_eval: int,
    output_name: str = '',
    fig_size: Tuple[int, int] = (6, 6)
    ):
    
    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis("off")
    
    cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])
    cmap.set_bad(color='black')
    
    vmin, vmax = -1, 1
    heat_map = ax.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    fig.colorbar(heat_map, orientation="horizontal", cax=cax)

    output_path = f"{output_name}_att_heat_map_{min_eval}.png"
    
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

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