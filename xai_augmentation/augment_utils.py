import os, torch, pickle, json
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

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

def extract_class_mean_vectors(model, dl, device):
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
            for crop_idx in range(0, len(vis_features)):
                mean_vectors[target.item()] += vis_features[crop_idx].cpu().numpy()
            num_crops[target.item()] += ncrops
    
    for c in idx_to_c.keys(): mean_vectors[c] /= num_crops[c]
    
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

        crop.save(f"{root_dir}/crops_for_augmentation/{c}/{instance_name[:-4]}_crop{i}.jpg")

def extract_augmented_crops(root_dir, xai_exp_dir, c, c_to_idx, mean_vector, n_augmentations, crop_size, patch_width, patch_height, mean_, std_, model, device):
    os.makedirs(f"{root_dir}/crops_for_augmentation/{c}", exist_ok=True)
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
                    
                    bottom_b_patch, right_b_patch = bottom_b_patch + 1, right_b_patch + 1
                    left_b_crop, top_b_crop, right_b_crop, bottom_b_crop = 0, 0, 0, 0
                    left_b_crop_to_pad, top_b_crop_to_pad, right_b_crop_to_pad, bottom_b_crop_to_pad = 0, 0, 0, 0
                    
                    if left_b_patch - 165 >= 0: left_b_crop = left_b_patch - 165
                    else:
                        left_b_crop = 0
                        left_b_crop_to_pad = 165 - left_b_patch
                    
                    if top_b_patch - 165 >= 0: top_b_crop = top_b_patch - 165
                    else:
                        top_b_crop = 0
                        top_b_crop_to_pad = 165 - top_b_patch
                    
                    if right_b_patch + 165 <= 902: right_b_crop = right_b_patch + 165
                    else:
                        right_b_crop = 902
                        right_b_crop_to_pad = 165 - (902 - right_b_patch)
                    
                    if bottom_b_patch + 165 <= 1279: bottom_b_crop = bottom_b_patch + 165
                    else:
                        bottom_b_crop = 1279
                        bottom_b_crop_to_pad = 165 - (1279 - bottom_b_patch)
                    
                    crop = img.crop((left_b_crop, top_b_crop, right_b_crop, bottom_b_crop))
                    mean_int = tuple(int(m * 255) for m in mean_)
                    crop = T.Pad((left_b_crop_to_pad, top_b_crop_to_pad, right_b_crop_to_pad, bottom_b_crop_to_pad), fill=mean_int)(crop)
                    
                    tensor_crop = T.ToTensor()(crop)
                    normalized_tensor_crop = T.Normalize(mean_, std_)(tensor_crop)
                    normalized_tensor_crop = normalized_tensor_crop.to(device)
                    
                    with torch.no_grad():
                        vis_f = model.extract_visual_features(normalized_tensor_crop.unsqueeze(0)).cpu().numpy()
                        out = model(normalized_tensor_crop.unsqueeze(0))
                        probs = F.softmax(out, dim=1).squeeze(0).cpu().numpy()
                        
                    cosine = np.dot(vis_f, mean_vector) / (np.linalg.norm(vis_f) * np.linalg.norm(mean_vector))
                    utility = np.exp(1-probs[label]) * cosine
                    
                    results.append({
                        "instance": name,
                        "utility": float(utility),
                        "patch_coordinates": tuple(map(int, (left_b_patch, top_b_patch, right_b_patch, bottom_b_patch))),
                        "crop_coordinates": tuple(map(int, (left_b_crop, top_b_crop, right_b_crop, bottom_b_crop))),
                        "pad_coordinates": tuple(map(int, (left_b_crop_to_pad, top_b_crop_to_pad, right_b_crop_to_pad, bottom_b_crop_to_pad)))
                    })
    
    results = sorted(results, key=lambda x: x['utility'], reverse=True)
    with open(f"{root_dir}/crops_for_augmentation/{c}_crops_for_augmentations.json", "w") as f:
        json.dump(results, f, indent=4)
    
    results = results[:n_augmentations]
    produce_augmented_crops(root_dir, results, c, mean_)