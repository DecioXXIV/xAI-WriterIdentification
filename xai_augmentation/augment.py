import os, torch, pickle
import numpy as np
import pandas as pd
import torchvision.transforms as T
import torch.nn.functional as F
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

from utils import load_metadata, get_train_instance_patterns

from classifiers.classifier_ResNet18.model import load_resnet18_classifier
from classifiers.classifier_ResNet18.utils.dataloader_utils import load_rgb_mean_std, Train_Test_DataLoader
from classifiers.classifier_ResNet18.utils.testing_utils import process_test_set

from xai.utils.explanations_utils import reduce_scores

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
CLASSIFIERS_ROOT = "./classifiers"
XAI_ROOT = "./xai"
XAI_AUG_ROOT = "./xai_augmentation"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase"])
    
    return parser.parse_args()

def load_model(model_type, num_classes, cp_base, test_id, exp_metadata, device):
    model, last_cp = None, None
    if model_type == "ResNet18":
        model, last_cp = load_resnet18_classifier(num_classes, "frozen", cp_base, "test", test_id, exp_metadata, device)
    
    return model, last_cp

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
    target_names = list(dataset.class_to_idx.keys())
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

def compute_augmentation_rates(model, dl, device):
    _, labels, preds, _, idx_to_c = process_test_set(dl, device, model)
    labels, preds = np.array(labels), np.array(preds)
    
    augmentations = dict()
    for idx in idx_to_c.keys():
        num_instances = np.argwhere(labels == idx).shape[0]
        num_correct = np.argwhere((labels == idx) & (labels == preds)).shape[0]
        
        aug_rate = 1 - (num_correct / num_instances)
        augmentations[idx] = int(aug_rate * 16 * num_instances)
            
    return augmentations

def produce_augmented_crops(root_dir, test_id, xai_exp_dir, c, c_to_idx, mean_vector, n_augmentations, mean_, std_, model, device):
    os.makedirs(f"{root_dir}/crops_for_augmentation/{c}", exist_ok=True)
    instance_names = os.listdir(f"{root_dir}/train_instances/{c}")
    mask = Image.open(f"{XAI_ROOT}/def_mask_{B_WIDTH}x{B_HEIGHT}.png")
    mask_array = np.array(mask)
    label = c_to_idx[str(c)]
    
    df = pd.DataFrame(columns=["instance", "utility", "patch_coordinates", "crop_coordinates", "pad_coordinates"])
    
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
                    
                    if ((bottom_b_patch - top_b_patch + 1)) != B_HEIGHT and (right_b_patch - left_b_patch + 1) != B_WIDTH:
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
                    
                    df.loc[len(df)] = [name, utility,
                                       (left_b_patch, top_b_patch, right_b_patch, bottom_b_patch),
                                       (left_b_crop, top_b_crop, right_b_crop, bottom_b_crop),
                                       (left_b_crop_to_pad, top_b_crop_to_pad, right_b_crop_to_pad, bottom_b_crop_to_pad)]
                    
    df = df.sort_values(by="utility", ascending=False)
    df = df.head(n_augmentations)
    df.to_csv(f"{root_dir}/crops_for_augmentation/{c}_crops_for_augmentations.csv", index=False, header=0)
                        
### #### ###
### MAIN ###
### #### ###
if __name__ == "__main__":
    args = get_args()
    
    TEST_ID = args.test_id
    XAI_ALGORITHM = args.xai_algorithm
    EXP_METADATA = load_metadata(f"{LOG_ROOT}/{TEST_ID}-metadata.json")
    DATASET = EXP_METADATA["DATASET"]
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]
    CLASSES_DATA = EXP_METADATA["CLASSES"]
    CLASSES = list(CLASSES_DATA.keys())
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CP = f"{CLASSIFIERS_ROOT}/cp/Test_3_TL_val_best_model.pth"
    
    model, _ = load_model(MODEL_TYPE, len(CLASSES_DATA), CP, TEST_ID, EXP_METADATA, DEVICE)
        
    root_dir = f"{XAI_AUG_ROOT}/{TEST_ID}"
    os.makedirs(root_dir, exist_ok=True)
    retrieve_training_instances(root_dir, DATASET, CLASSES)
    EXP_DIR = f"{CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/tests/{TEST_ID}"
    mean_, std_ = load_rgb_mean_std(EXP_DIR)
    dl = Train_Test_DataLoader(directory=f"{root_dir}/train_instances", classes=CLASSES, batch_size=1, img_crop_size=380, weighted_sampling=False, phase="test", mean=mean_, std=std_, shuffle=False)
    
    print("Extracting class mean vectors...")
    mean_class_vectors, c_to_idx, idx_to_c = extract_class_mean_vectors(model, dl, DEVICE)
    
    print("Computing augmentation rates...")
    augmentations = compute_augmentation_rates(model, dl, DEVICE)
    
    os.makedirs(f"{root_dir}/crops_for_augmentation", exist_ok=True)
    
    XAI_EXP_DIR_NAME = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["DIR_NAME"]
    B_WIDTH, B_HEIGHT = EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["BLOCK_DIM"]["WIDTH"], EXP_METADATA[f"{XAI_ALGORITHM}_base_METADATA"]["BLOCK_DIM"]["HEIGHT"]
    XAI_EXP_DIR = f"{XAI_ROOT}/explanations/patches_{B_WIDTH}x{B_HEIGHT}_removal/{XAI_EXP_DIR_NAME}"
    for c in CLASSES:
        print(f"Producing Augmentation crops for class: {c}")
        mean_class_vector = mean_class_vectors[c_to_idx[str(c)]]
        n_augmentations = augmentations[c_to_idx[str(c)]]
        produce_augmented_crops(root_dir, TEST_ID, XAI_EXP_DIR, c, c_to_idx, mean_class_vector, n_augmentations, mean_, std_, model, DEVICE)