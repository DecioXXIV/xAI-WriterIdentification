import torch, pickle, cv2, os, pathlib, glob, random
from PIL import Image
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import torchvision.transforms as T
from captum.attr import Lime
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import visualization as viz
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from typing import Tuple
from create_image_grid import create_image_grid
from model_utils import Classification_Model
from image_utils import apply_transforms_crops, load_rgb_mean_std
import matplotlib.pyplot as plt

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

def get_rois(scores_matrix, page, mask, mask_size, pagename, num_rois:int = None, threshold = 0.5):

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
    
    cv2.imwrite(f'{output_dir}{pagename}_rois_t_{str(threshold)}.png', im_rgb)
    
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
        crop = im_rgb_backup[roi_matrix == idx].reshape(mask_size[0],mask_size[1],3)
        cv2.imwrite(f'{output_dir}{pagename}_ROI_{str(k)}.png', crop)

mask_path = './explanations/page_level/Vat.lat.653_0011_fr_0001r_m/Vat.lat.653_0011_fr_0001r_m_mask_blocks_50x50.png'
cp_base = "./../classifier_NN/cp/Test_3_TL_val_best_model.pth"
cp = "./../classifier_NN/tests/output/VatLat653/checkpoints/Test_VatLat653_MLC_val_best_model.pth"
data_dir = './data/'
# output_dir = './output/'
output_dir = './explanations/page_level/Vat.lat.653_0011_fr_0001r_m/patches_50x50/'
n_iter = 5
overlap = 180
reduction_method = 'mean'
min_eval = 2
mask_size = (50, 50)

#mean, std = load_rgb_mean_std(f'{data_dir}train')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classification_Model(mode = 'frozen', cp_path = cp_base, num_classes = 4)
model = model.to(DEVICE)
model.load_state_dict(torch.load(cp)['model_state_dict'])
model.eval()
label_idx = 0
exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)

for img_path in tqdm(glob.glob('./data/*.jpg')[:1]):
    
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    scores_name = f'{img_name}_dict_scores.pickle'

    base_img = Image.open(img_path)
    base_mask = Image.open(mask_path)
    #label_idx = int(pathlib.Path(img_path).parts[-2])

    scores = defaultdict(list)  

    if not os.path.exists(f'{output_dir}{scores_name}'):

        G, nc, nr = create_image_grid(380, overlap, base_img)

        for i in tqdm(range(nr), leave = False):
            for j in tqdm(range(nc), leave = False):
                x0, y0, x1, y1 = G[f'{i}_{j}']
                img_crop = base_img.crop((x0, y0, x1, y1))
                mask_crop = base_mask.crop((x0, y0, x1, y1))
                mask_array = np.array(deepcopy(mask_crop))
                idxs = np.unique(mask_array)

                img, mask, n_interpret_features = apply_transforms_crops(img_crop, mask_crop, 0.5, 0.5)
                img = img.to(DEVICE)
                mask = mask.to(DEVICE)
                input_ = img.unsqueeze(0)
                feature_mask = mask.unsqueeze(0)

                attr_map_mean = np.zeros((380, 380, 3))

                for k in range(n_iter):

                    lr_lime = Lime(model,
                    interpretable_model = SkLearnLinearRegression(),
                    similarity_func = exp_eucl_distance
                    )

                    attrs = lr_lime.attribute(
                        input_,
                        target = label_idx,
                        feature_mask = feature_mask,
                        n_samples = 40,
                        perturbations_per_eval = 16,
                        show_progress = False
                    ).squeeze(0)

                    attr_map = attrs.permute(1, 2, 0).cpu().numpy()

                    attr_map_mean += attr_map

                attr_map_mean/=n_iter

                norm_attr = viz._normalize_image_attr(attr_map_mean, "all")
                norm_attr_3d = norm_attr[:, :, None] * np.ones(3, dtype=int)[None, None, :]

                for idx in idxs:
                    super_pixel = norm_attr_3d[mask_array == idx]
                    if len(super_pixel) == mask_size[0]*mask_size[1]:
                        scores[idx].append(np.mean(super_pixel))

        with open(f'{output_dir}{scores_name}', 'wb') as handle:
            pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        with open(f'{output_dir}{scores_name}', 'rb') as handle:
            scores = pickle.load(handle)

        simplified_scores = reduce_scores(base_mask, scores, reduction_method, min_eval)
        mask_with_attr_scores = assign_attr_scores_to_mask(base_mask, simplified_scores)
        custom_visualization(mask_with_attr_scores, min_eval, f'{output_dir}{img_name}')
        
        get_rois(mask_with_attr_scores, base_img, base_mask, mask_size, img_name, num_rois = 5, threshold = 0.5)