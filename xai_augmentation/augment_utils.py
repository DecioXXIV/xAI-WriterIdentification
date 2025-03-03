import os, torch, pickle, json
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA

from utils import get_train_instance_patterns

from classifiers.utils.testing_utils import process_test_set

from xai.utils.explanations_utils import reduce_scores

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
CLASSIFIERS_ROOT = "./classifiers"
XAI_ROOT = "./xai"
XAI_AUG_ROOT = "./xai_augmentation"

def retrieve_training_instances(root_dir, dataset, classes):
    os.makedirs(f"{root_dir}/train_instances", exist_ok=True)
    dataset_dir = f"{DATASET_ROOT}/{dataset}"
    train_instance_patterns = get_train_instance_patterns()
    
    for c in classes:
        os.makedirs(f"{root_dir}/train_instances/{c}", exist_ok=True)
        train_instances = [f for f in os.listdir(f"{dataset_dir}/processed/{c}") if train_instance_patterns[dataset](f)]
        for f in train_instances:
            source, dest = f"{dataset_dir}/processed/{c}/{f}", f"{root_dir}/train_instances/{c}/{f}"
            os.system(f"cp {source} {dest}")

def setup_dimensionality_reduction(model, dl, device, final_dim):
    pca, pca_input = PCA(n_components=final_dim), None
    dataset, set_ = dl.load_data()
    
    total_crops, crop_counter = None, 0
    for data, _ in tqdm(set_, desc="Processing Training Instances to setup PCA-dimensionality reducer"):
        data = data.to(device)
        _, ncrops, c, h, w = data.size()

        with torch.no_grad():
            vis_features = model.extract_visual_features(data.view(-1, c, h, w))
            if pca_input is None:
                total_crops = len(dataset) * ncrops
                pca_input = np.zeros((total_crops, vis_features.size()[1]))
                
            for crop_idx in range(0, vis_features.size()[0]):
                pca_input[crop_counter] = vis_features[crop_idx].cpu().numpy()
                crop_counter += 1
    
    pca.fit(pca_input)

    return pca
    
def extract_class_mean_vectors(model, dl, pca, device):
    dataset, set_ = dl.load_data()
    c_to_idx = dataset.class_to_idx
    idx_to_c = {c_to_idx[k]: k for k in list(c_to_idx.keys())}
    
    mean_vectors, num_crops = dict(), dict()
    for c in idx_to_c.keys(): mean_vectors[c], num_crops[c] = np.zeros(512), 0
    
    for data, target in tqdm(set_):
        data = data.to(device)
        _, ncrops, c, h, w = data.size()
        
        with torch.no_grad():
            vis_features = model.extract_visual_features(data.view(-1, c, h, w))
            for crop_idx in range(0, vis_features.size()[0]):
                mean_vectors[target.item()] += vis_features[crop_idx].cpu().numpy()
            num_crops[target.item()] += ncrops
    
    for c in idx_to_c.keys(): mean_vectors[c] /= num_crops[c]

    if pca is not None:
        for c in idx_to_c.keys():
            reduced_mean_vector = pca.transform(mean_vectors[c].reshape(1, -1))
            mean_vectors[c] = reduced_mean_vector[0]

    return mean_vectors, c_to_idx, idx_to_c

def compute_augmentation_rates(model, dl, idx_to_c, device):
    ITERS = 5
    augmentations = {idx: np.zeros(ITERS) for idx in idx_to_c.keys()}

    for it in range(0, ITERS):
        _, labels, preds, _, idx_to_c = process_test_set(dl, device, model)
        labels, preds = np.array(labels), np.array(preds)
    
        for idx in idx_to_c.keys():
            num_instances = np.sum(labels == idx)
            num_correct = np.sum((labels == idx) & (labels == preds))
            aug_rate = 1 - (num_correct / num_instances)
            augmentations[idx][it] = int(aug_rate * 16 * num_instances)

    for idx in idx_to_c.keys(): augmentations[idx] = int(np.max(augmentations[idx]))

    return augmentations

def produce_augmented_crops(root_dir, results, c, mean_):
    for i, row in enumerate(results):
        instance_name = row["instance"]
        img = Image.open(f"{root_dir}/train_instances/{c}/{instance_name}")

        crop_coordinates = row["crop_coordinates"]
        pad_coordinates = row["pad_coordinates"]

        crop = img.crop(crop_coordinates)
        mean_int = tuple(int(m * 255) for m in mean_)
        crop = T.Pad(pad_coordinates, fill=mean_int)(crop)

        crop.save(f"{root_dir}/crops_for_augmentation/{c}/{instance_name[:-4]}_crop{i}{instance_name[-4:]}")

def produce_random_augmented_crops(root_dir, c, crop_size, random_augmentations):
    instance_names = os.listdir(f"{root_dir}/train_instances/{c}")
    crops2instance = {k: 0 for k in instance_names}
    
    while random_augmentations > 0:
        for k in instance_names:
            crops2instance[k] += 1
            random_augmentations -= 1
            if random_augmentations == 0: break
    
    for k in instance_names:
        img = Image.open(f"{root_dir}/train_instances/{c}/{k}")
        img_w, img_h = img.size
        crops_to_extract = crops2instance[k]
        for i in range(0, crops_to_extract):
            left = np.random.randint(low=0, high=img_w - crop_size + 1)
            top = np.random.randint(low=0, high=img_h - crop_size + 1)
            right, bottom = left + crop_size, top + crop_size
            
            crop = img.crop((left, top, right, bottom))
            crop.save(f"{root_dir}/crops_for_augmentation/{c}/{k[:-4]}_random_crop{i}{k[-4:]}")
            
def get_crop_from_patch(crop_size, patch_width, patch_height, left_b_patch, top_b_patch, right_b_patch, bottom_b_patch):
    left_b_crop, top_b_crop, right_b_crop, bottom_b_crop = 0, 0, 0, 0
    left_b_crop_to_pad, top_b_crop_to_pad, right_b_crop_to_pad, bottom_b_crop_to_pad = 0, 0, 0, 0
    
    crop_left_top_semisize, crop_right_bottom_semisize = 0, 0
    if crop_size % 2 == 0: crop_left_top_semisize, crop_right_bottom_semisize = int(crop_size/2), int(crop_size/2)
    else: crop_left_top_semisize, crop_right_bottom_semisize = int(crop_size/2), int(crop_size/2)+1
    
    patch_left_semisize, patch_right_semisize = 0, 0
    if patch_width % 2 == 0: patch_left_semisize, patch_right_semisize = int(patch_width/2), int(patch_width/2)
    else: patch_left_semisize, patch_right_semisize = int(patch_width/2), int(patch_width/2)+1
    
    patch_top_semisize, patch_bottom_semisize = 0, 0
    if patch_height % 2 == 0: patch_top_semisize, patch_bottom_semisize = int(patch_height/2), int(patch_height/2)
    else: patch_top_semisize, patch_bottom_semisize = int(patch_height/2), int(patch_height/2)+1
    
    left_difference = crop_left_top_semisize - patch_left_semisize
    top_difference = crop_left_top_semisize - patch_top_semisize
    right_difference = crop_right_bottom_semisize - patch_right_semisize
    bottom_difference = crop_right_bottom_semisize - patch_bottom_semisize
    
    if left_b_patch - left_difference >= 0: left_b_crop = left_b_patch - left_difference
    else: left_b_crop_to_pad = left_difference - left_b_patch
    
    if top_b_patch - top_difference >= 0: top_b_crop = top_b_patch - top_difference
    else: top_b_crop_to_pad = top_difference - top_b_patch
    
    if right_b_patch + right_difference <= 902: right_b_crop = right_b_patch + right_difference
    else: 
        right_b_crop = 902
        right_b_crop_to_pad = right_difference - (902 - right_b_patch)
    
    if bottom_b_patch + bottom_difference <= 1279: bottom_b_crop = bottom_b_patch + bottom_difference
    else: 
        bottom_b_crop = 1279
        bottom_b_crop_to_pad = bottom_difference - (1279 - bottom_b_patch)
    
    return (left_b_crop, top_b_crop, right_b_crop, bottom_b_crop), (left_b_crop_to_pad, top_b_crop_to_pad, right_b_crop_to_pad, bottom_b_crop_to_pad)

def extract_augmented_crops_protect_inform(root_dir, xai_exp_dir, c, c_to_idx, mean_vectors, crop_size, patch_width, patch_height, mean_, std_, pca, model, device):  
    instance_names = os.listdir(f"{root_dir}/train_instances/{c}")
    mask = Image.open(f"{XAI_ROOT}/def_mask_{patch_width}x{patch_height}.png")
    mask_array = np.array(mask)
    label = c_to_idx[str(c)]
    mean_class_vector = mean_vectors[c_to_idx[str(c)]]

    results = []
    
    for name in tqdm(instance_names):
        img = Image.open(f"{root_dir}/train_instances/{c}/{name}")
        inst_name = name[:-4]
        with open(f"{xai_exp_dir}/{inst_name}/{inst_name}_scores.pkl", "rb") as f:
            base_scores = pickle.load(f)
        reduced_scores = reduce_scores(mask, base_scores)
        
        for idx, score in reduced_scores.items():
            if score != [np.nan] and score < 0:
                positions = np.argwhere(mask_array == idx)
                if positions.size > 0:
                    top_b_patch, left_b_patch = positions.min(axis=0)
                    bottom_b_patch, right_b_patch = positions.max(axis=0)
                    
                    if ((bottom_b_patch - top_b_patch + 1)) != patch_height and (right_b_patch - left_b_patch + 1) != patch_width:
                        print(f"Skipped patch {idx} for instance {inst_name} because of wrong dimensions")
                        idx += 1
                        continue
                    
                    right_b_patch, bottom_b_patch = right_b_patch + 1, bottom_b_patch + 1
                    crop_coordinates, pad_coordinates = get_crop_from_patch(crop_size, patch_width, patch_height, left_b_patch, top_b_patch, right_b_patch, bottom_b_patch)
                    
                    crop = img.crop(crop_coordinates)
                    mean_int = tuple(int(m * 255) for m in mean_)
                    crop = T.Pad(pad_coordinates, fill=mean_int)(crop)
                    
                    tensor_crop = T.ToTensor()(crop)
                    normalized_tensor_crop = T.Normalize(mean_, std_)(tensor_crop)
                    normalized_tensor_crop = normalized_tensor_crop.to(device)
                    
                    with torch.no_grad():
                        vis_f = model.extract_visual_features(normalized_tensor_crop.unsqueeze(0)).cpu().numpy()
                        out = model(normalized_tensor_crop.unsqueeze(0))
                        probs = F.softmax(out, dim=1).squeeze(0).cpu().numpy()
                    
                    vis_f = pca.transform(vis_f.reshape(1, -1))[0]
                        
                    cosine = np.dot(vis_f, mean_class_vector) / (np.linalg.norm(vis_f) * np.linalg.norm(mean_class_vector))
                    pi_utility = np.exp(1-probs[label]) * cosine
                    
                    results.append({
                        "instance": name,
                        "utility": float(pi_utility),
                        "patch_coordinates": tuple(map(int, (left_b_patch, top_b_patch, right_b_patch, bottom_b_patch))),
                        "crop_coordinates": tuple(map(int, crop_coordinates)),
                        "pad_coordinates": tuple(map(int, pad_coordinates))
                    })
    
    return results

def extract_augmented_crops_lime_reds(root_dir, xai_exp_dir, c, crop_size, patch_width, patch_height):    
    instance_names = os.listdir(f"{root_dir}/train_instances/{c}")
    mask = Image.open(f"{XAI_ROOT}/def_mask_{patch_width}x{patch_height}.png")
    mask_array = np.array(mask)

    results = []

    for name in tqdm(instance_names):
        img = Image.open(f"{root_dir}/train_instances/{c}/{name}")
        inst_name = name[:-4]
        with open(f"{xai_exp_dir}/{inst_name}/{inst_name}_scores.pkl", "rb") as f:
            base_scores = pickle.load(f)
        reduced_scores = reduce_scores(mask, base_scores)
        
        for idx, score in reduced_scores.items():
            if score != [np.nan] and score < 0:
                positions = np.argwhere(mask_array == idx)
                if positions.size > 0:
                    top_b_patch, left_b_patch = positions.min(axis=0)
                    bottom_b_patch, right_b_patch = positions.max(axis=0)
                    
                    if ((bottom_b_patch - top_b_patch + 1)) != patch_height and (right_b_patch - left_b_patch + 1) != patch_width:
                        print(f"Skipped patch {idx} for instance {inst_name} because of wrong dimensions")
                        idx += 1
                        continue
                    
                    right_b_patch, bottom_b_patch = right_b_patch + 1, bottom_b_patch + 1
                    crop_coordinates, pad_coordinates = get_crop_from_patch(crop_size, patch_width, patch_height, left_b_patch, top_b_patch, right_b_patch, bottom_b_patch)
                    
                    left_b_crop, top_b_crop, right_b_crop, bottom_b_crop = crop_coordinates
                    mask_crop = mask_array[left_b_crop:right_b_crop+1, top_b_crop:bottom_b_crop+1]
                    
                    crop_area = (right_b_crop - left_b_crop) * (bottom_b_crop - top_b_crop)
                    mask_crop_idxs = np.unique(mask_crop)
                    total_attribution_score = 0
                    
                    for elem in mask_crop_idxs:
                        idx_area = np.sum(mask_crop_idxs == elem)
                        if reduced_scores[elem] != [np.nan]: total_attribution_score += reduced_scores[elem] * idx_area
                        else: continue
                        
                    lr_utility = -1 * total_attribution_score / crop_area
                    
                    results.append({
                        "instance": name,
                        "utility": float(lr_utility),
                        "patch_coordinates": tuple(map(int, (left_b_patch, top_b_patch, right_b_patch, bottom_b_patch))),
                        "crop_coordinates": tuple(map(int, crop_coordinates)),
                        "pad_coordinates": tuple(map(int, pad_coordinates))
                    })
    
    return results

def extract_augmented_crops_world_opening(root_dir, xai_exp_dir, c, c_to_idx, mean_vectors, crop_size, patch_width, patch_height, mean_, std_, pca, model, device):
    instance_names = os.listdir(f"{root_dir}/train_instances/{c}")
    mask = Image.open(f"{XAI_ROOT}/def_mask_{patch_width}x{patch_height}.png")
    mask_array = np.array(mask)
    label = c_to_idx[str(c)]

    results = []
    
    for name in tqdm(instance_names):
        img = Image.open(f"{root_dir}/train_instances/{c}/{name}")
        inst_name = name[:-4]
        with open(f"{xai_exp_dir}/{inst_name}/{inst_name}_scores.pkl", "rb") as f:
            base_scores = pickle.load(f)
        reduced_scores = reduce_scores(mask, base_scores)
        
        for idx, score in reduced_scores.items():
            if score != [np.nan] and score < 0:
                positions = np.argwhere(mask_array == idx)
                if positions.size > 0:
                    top_b_patch, left_b_patch = positions.min(axis=0)
                    bottom_b_patch, right_b_patch = positions.max(axis=0)
                    
                    if ((bottom_b_patch - top_b_patch + 1)) != patch_height and (right_b_patch - left_b_patch + 1) != patch_width:
                        print(f"Skipped patch {idx} for instance {inst_name} because of wrong dimensions")
                        idx += 1
                        continue
                    
                    right_b_patch, bottom_b_patch = right_b_patch + 1, bottom_b_patch + 1
                    crop_coordinates, pad_coordinates = get_crop_from_patch(crop_size, patch_width, patch_height, left_b_patch, top_b_patch, right_b_patch, bottom_b_patch)
                    
                    crop = img.crop(crop_coordinates)
                    mean_int = tuple(int(m * 255) for m in mean_)
                    crop = T.Pad(pad_coordinates, fill=mean_int)(crop)
                    
                    tensor_crop = T.ToTensor()(crop)
                    normalized_tensor_crop = T.Normalize(mean_, std_)(tensor_crop)
                    normalized_tensor_crop = normalized_tensor_crop.to(device)
                    
                    with torch.no_grad(): vis_f = model.extract_visual_features(normalized_tensor_crop.unsqueeze(0)).cpu().numpy()
                    vis_f = pca.transform(vis_f.reshape(1, -1))[0]
                    
                    euc_different_classes = 0
                    for k in mean_vectors.keys():
                        if k != label: euc_different_classes += np.linalg.norm(vis_f - mean_vectors[k])
                    mean_euc_different_classes = euc_different_classes / (len(mean_vectors) - 1)
                    euc_true_class = np.linalg.norm(vis_f - mean_vectors[label])
                    wo_utility = mean_euc_different_classes - euc_true_class
                    
                    results.append({
                        "instance": name,
                        "utility": float(wo_utility),
                        "patch_coordinates": tuple(map(int, (left_b_patch, top_b_patch, right_b_patch, bottom_b_patch))),
                        "crop_coordinates": tuple(map(int, crop_coordinates)),
                        "pad_coordinates": tuple(map(int, pad_coordinates))
                    })
    
    return results
        
def extract_augmented_crops(mode, root_dir, xai_exp_dir, c, c_to_idx, mean_vectors, xai_and_random_augmentations, crop_size, patch_width, patch_height, mean_, std_, pca, model, device):
    os.makedirs(f"{root_dir}/crops_for_augmentation/{c}", exist_ok=True)
    xai_augmentations, random_augmentations = xai_and_random_augmentations
    
    if xai_augmentations > 0:
        print(f"Extracting {xai_augmentations} XAI-Augmentation Crops for class {c}")
        results = None
        if mode == "pi": results = extract_augmented_crops_protect_inform(root_dir, xai_exp_dir, c, c_to_idx, mean_vectors, crop_size, patch_width, patch_height, mean_, std_, pca, model, device)
        elif mode == "lr": results = extract_augmented_crops_lime_reds(root_dir, xai_exp_dir, c, crop_size, patch_width, patch_height)
        elif mode == "wo": results = extract_augmented_crops_world_opening(root_dir, xai_exp_dir, c, c_to_idx, mean_vectors, crop_size, patch_width, patch_height, mean_, std_, pca, model, device)
    
        results = sorted(results, key=lambda x: x['utility'], reverse=True)
        with open(f"{root_dir}/crops_for_augmentation/{c}_crops_for_augmentations.json", "w") as f:
            json.dump(results, f, indent=4)
    
        results = results[:xai_augmentations]
        produce_augmented_crops(root_dir, results, c, mean_)
    else: print(f"Skipping extraction of XAI-Augmentation Crops for class {c}: no Crops have to be extracted in this way")
    
    if random_augmentations > 0:
        print(f"Extracting {random_augmentations} Random-Augmentation Crops for class {c}")
        produce_random_augmented_crops(root_dir, c, crop_size, random_augmentations)
    
    else: print(f"Skipping extraction of Random-Augmentation Crops for class {c}: no Crops have to be extracted in this way")