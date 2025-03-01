import os, torch, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn import functional as F
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import load_metadata, get_test_instance_patterns, get_model_base_checkpoint

from classifiers.utils.fine_tune_utils import load_model
from classifiers.classifier_ResNet18.utils.dataloader_utils import Confidence_Test_DataLoader, load_rgb_mean_std
from classifiers.classifier_ResNet18.utils.testing_utils import produce_confusion_matrix

from xai.utils.explanations_utils import reduce_scores, assign_attr_scores_to_mask

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
CLASSIFIERS_ROOT = "./classifiers"
XAI_ROOT = "./xai"
EVALS_ROOT = "./evals"

def retrieve_test_instances(root_dir, dataset, classes):
    os.makedirs(f"{root_dir}/test_instances", exist_ok=True)
    dataset_dir = f"./datasets/{dataset}"
    test_instance_patterns = get_test_instance_patterns()
    
    for c in classes:
        os.makedirs(f"{root_dir}/test_instances/{c}", exist_ok=True)
        
        test_instances = [f for f in os.listdir(f"{dataset_dir}/processed/{c}") if test_instance_patterns[dataset](f)]
        for f in test_instances:
            source, dest = f"{dataset_dir}/processed/{c}/{f}", f"{root_dir}/test_instances/{c}/{f}"
            os.system(f"cp {source} {dest}")

### ################### ###
### PRINCIPAL FUNCTIONS ###
### ################### ###
def pair_confidence_test(baseline_id, retrained_id, mode, mult_factors, iters):
    BASELINE_METADATA = load_metadata(f"{LOG_ROOT}/{baseline_id}-metadata.json")
    RETRAINED_METADATA = load_metadata(f"{LOG_ROOT}/{retrained_id}-metadata.json")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CROP_SIZE = 380
    B_MODEL_TYPE, R_MODEL_TYPE = BASELINE_METADATA["MODEL_TYPE"], RETRAINED_METADATA["MODEL_TYPE"]
    BASELINE_CLASSES, RETRAINED_CLASSES = BASELINE_METADATA["CLASSES"], RETRAINED_METADATA["CLASSES"]
    cp_base = get_model_base_checkpoint(B_MODEL_TYPE)
    
    model_baseline, _ = load_model(B_MODEL_TYPE, len(BASELINE_CLASSES), "frozen", cp_base, "test", baseline_id, BASELINE_METADATA, DEVICE)
    model_retrained, _ = load_model(R_MODEL_TYPE, len(RETRAINED_CLASSES), "frozen", cp_base, "test", retrained_id, RETRAINED_METADATA, DEVICE)
    
    root_dir = f"{EVALS_ROOT}/bvr_comparisons/{baseline_id}/VERSUS-{retrained_id}/confidence/{mode}"
    os.makedirs(root_dir, exist_ok=True)
    
    DATASET = BASELINE_METADATA["DATASET"]
    CLASSES = list(BASELINE_METADATA["CLASSES"].keys())
    mean_, std_ = load_rgb_mean_std(f"{CLASSIFIERS_ROOT}/classifier_{B_MODEL_TYPE}/tests/{baseline_id}")
    
    retrieve_test_instances(root_dir, DATASET, CLASSES)
    test_instances = list()
    for c in CLASSES: test_instances.extend(os.listdir(f"{root_dir}/test_instances/{c}"))
    
    if mode == "scan": 
        for mf in mult_factors: 
            dl = Confidence_Test_DataLoader(f"{root_dir}/test_instances", CLASSES, 1, CROP_SIZE, False, "scan", mf, mean_, std_, False)
            execute_pair_confidence_test(model_baseline, model_retrained, dl, test_instances, DEVICE, root_dir, CLASSES, mode, f, None)
    if mode == "random":
        dl = Confidence_Test_DataLoader(f"{root_dir}/test_instances", CLASSES, 1, CROP_SIZE, False, "random", 1, mean_, std_, False)
        for iter in range(0, iters): 
            execute_pair_confidence_test(model_baseline, model_retrained, dl, test_instances, DEVICE, root_dir, CLASSES, mode, None, iter+1)
    
    os.system(f"rm -r {root_dir}/test_instances")

def pair_explanations_test(baseline_id, retrained_id, xai_algorithm):
    BASELINE_METADATA = load_metadata(f"{LOG_ROOT}/{baseline_id}-metadata.json")
    RETRAINED_METADATA = load_metadata(f"{LOG_ROOT}/{retrained_id}-metadata.json")

    B_EXP_DIR = BASELINE_METADATA[f"{xai_algorithm}_METADATA"]["DIR_NAME"]
    R_EXP_DIR = RETRAINED_METADATA[f"{xai_algorithm}_METADATA"]["DIR_NAME"]

    PATCH_W = BASELINE_METADATA[f"{xai_algorithm}_METADATA"]["PATCH_DIM"]["WIDTH"]
    PATCH_H = BASELINE_METADATA[f"{xai_algorithm}_METADATA"]["PATCH_DIM"]["HEIGHT"]

    root_dir = f"{EVALS_ROOT}/bvr_comparisons/{baseline_id}/VERSUS-{retrained_id}/explanations/{xai_algorithm}"
    os.makedirs(root_dir, exist_ok=True)

    test_instance_names = os.listdir(f"{XAI_ROOT}/explanations/patches_{PATCH_W}x{PATCH_H}_removal/{R_EXP_DIR}")
    test_instance_names.remove("rgb_train_stats.pkl")

    for name in test_instance_names:
        base_scores_path = f"{XAI_ROOT}/explanations/patches_{PATCH_W}x{PATCH_H}_removal/{B_EXP_DIR}/{name}/{name}_scores.pkl"
        ret_scores_path = f"{XAI_ROOT}/explanations/patches_{PATCH_W}x{PATCH_H}_removal/{R_EXP_DIR}/{name}/{name}_scores.pkl"

        base_scores, ret_scores = None, None
        with open(base_scores_path, "rb") as f: base_scores = pickle.load(f)
        with open(ret_scores_path, "rb") as f: ret_scores = pickle.load(f)

        mask = Image.open(f"{XAI_ROOT}/def_mask_{PATCH_W}x{PATCH_H}.png")

        base_scores_reduced, ret_scores_reduced = reduce_scores(mask, base_scores), reduce_scores(mask, ret_scores)
        b_mask_with_scores, r_mask_with_scores  = assign_attr_scores_to_mask(mask, base_scores_reduced), assign_attr_scores_to_mask(mask, ret_scores_reduced)

        produce_explanation_comparison(root_dir, name, b_mask_with_scores, r_mask_with_scores)
        print(f"Processed Test Instance: {name}")

### ############################## ###
### PAIR CONFIDENCE TEST FUNCTIONS ###
### ############################## ###
def execute_pair_confidence_test(model_b, model_r, dl, instances, device, root_dir, classes, mode, mult_factor=None, iter=None):
    if mode == "scan": print(f"Executing '{mode}' Confidence Pair Test with 'mult_factor' = {mult_factor}")
    if mode == "random": print(f"Executing Iteration {iter} of '{mode}' Confidence Pair Test")
    
    dataset_, set_ = dl.load_data()
    target_names = list(dataset_.class_to_idx.keys())
    c_to_idx = dataset_.class_to_idx
    idx_to_c = {c_to_idx[k]: k for k in list(c_to_idx.keys())}
    
    labels, preds_b, preds_r = models_inference((model_b, model_r), set_, instances, device, root_dir, classes, mode, mult_factor, iter)
    produce_delta_reports(instances, root_dir, mode, mult_factor, iter)
    produce_comparison_reports(idx_to_c, labels, preds_b, preds_r, target_names, instances, root_dir, mode, mult_factor, iter)

def models_inference(models, dataset, instances, device, root_dir, classes, mode, mult_factor, iter):
    torch.backends.cudnn.benchmark = True
    
    labels, preds_b, preds_r = list(), list(), list()
    columns = ["Instance", "Crop_No"]
    for c in classes: columns.append(f"P{c}_Baseline")
    for c in classes: columns.append(f"P{c}_Ret")
    columns.extend(["True_Label", "Pred_Baseline", "Pred_Ret"])
    
    df_probs = pd.DataFrame(columns=columns)
    j = 0
    
    for data, target in tqdm(dataset):
        labels += list(target.numpy())
        data, target = data.to(device), target.to(device)
        bs, ncrops, channels, h, w = data.size()
        model_b, model_r = models
        
        with torch.no_grad():
            out_b, out_r = model_b(data.view(-1, channels, h, w)), model_r(data.view(-1, channels, h, w))
            out_b, out_r = F.softmax(out_b, dim=1), F.softmax(out_r, dim=1)
            
            max_index_b = out_b.max(dim=1)[1].cpu().detach().numpy()
            max_index_r = out_r.max(dim=1)[1].cpu().detach().numpy()
            
            final_max_index_b, final_max_index_r = list(), list()
            
            max_index_over_crops_b = max_index_b.reshape(bs, ncrops)
            max_index_over_crops_r = max_index_r.reshape(bs, ncrops)
            
            for s in range(0, bs):
                final_max_index_b.append(np.argmax(np.bincount(max_index_over_crops_b[s, :])))
                final_max_index_r.append(np.argmax(np.bincount(max_index_over_crops_r[s, :])))
            
            preds_b += list(final_max_index_b)
            preds_r += list(final_max_index_r)
            
        for i in range(0, ncrops):
            values = [instances[j], i]
            for c_idx in range(0, len(classes)): values.append(out_b[i][c_idx].item())
            for c_idx in range(0, len(classes)): values.append(out_r[i][c_idx].item())

            values.extend([target.cpu().numpy()[0], max_index_b[i], max_index_r[i]])
            
            df_probs.loc[len(df_probs)] = values
        
        j += 1
        
    if mode == "scan":
        os.makedirs(f"{root_dir}/fact_{mult_factor}")
        df_probs.to_csv(f"{root_dir}/fact_{mult_factor}/comparison_probs.csv", index=False, header=True)
    if mode == "random":
        os.makedirs(f"{root_dir}/iter_{iter}")
        df_probs.to_csv(f"{root_dir}/iter_{iter}/comparison_probs.csv", index=False, header=True)
    
    return labels, preds_b, preds_r
            
def produce_delta_reports(instances, root_dir, mode, mult_factor, iter):
    df_stats = pd.DataFrame(columns=["Instance", "Baseline_Mean", "Baseline_Var", "Ret_Mean", "Ret_Var", "Delta_Mean"])
    
    df_probs = None
    if mode == "scan": df_probs = pd.read_csv(f"{root_dir}/fact_{mult_factor}/comparison_probs.csv", header=0)
    if mode == "random": df_probs = pd.read_csv(f"{root_dir}/iter_{iter}/comparison_probs.csv", header=0)
    
    for instance in instances:
        label = int(instance[0:4])
        
        stats = df_probs[(df_probs["Instance"] == instance)]
        
        baseline_mean = np.mean(stats[f"P{label}_Baseline"].tolist())
        baseline_var = np.var(stats[f"P{label}_Baseline"].tolist())
        ret_mean = np.mean(stats[f"P{label}_Ret"].tolist())
        ret_var = np.var(stats[f"P{label}_Ret"].tolist())
        
        delta_mean = baseline_mean - ret_mean
        
        df_stats.loc[len(df_stats)] = [instance, baseline_mean, baseline_var, ret_mean, ret_var, delta_mean]
    
    if mode == "scan": df_stats.to_csv(f"{root_dir}/fact_{mult_factor}/comparison_stats.csv", index=False, header=True)
    if mode == "random": df_stats.to_csv(f"{root_dir}/iter_{iter}/comparison_stats.csv", index=False, header=True)
    
def produce_comparison_reports(idx_to_c, labels, preds_b, preds_r, target_names, instances, root_dir, mode, mult_factor, iter):
    label_class_names = [idx_to_c[id_] for id_ in labels]
    pred_b_class_names = [idx_to_c[id_] for id_ in preds_b]
    pred_r_class_names = [idx_to_c[id_] for id_ in preds_r]
    
    df_preds = pd.DataFrame(columns=["Instance", "True_Label", "Pred_B_Label", "Pred_R_Label"])
    
    df_preds["Instance"] = instances
    df_preds["True_Label"] = label_class_names
    df_preds["Pred_B_Label"] = pred_b_class_names
    df_preds["Pred_R_Label"] = pred_r_class_names
    
    dir = None
    if mode == "scan": dir = f"{root_dir}/fact_{mult_factor}" 
    if mode == "random": dir = f"{root_dir}/iter_{iter}"
    
    df_preds.to_csv(f"{dir}/comparison_labels.csv", index=False, header=True)    
    produce_confusion_matrix(labels, preds_b, target_names, idx_to_c, dir, "baseline")
    produce_confusion_matrix(labels, preds_r, target_names, idx_to_c, dir, "retrained")

### ################################ ###
### PAIR EXPLANATIONS TEST FUNCTIONS ###
### ################################ ###
def produce_explanation_comparison(dir, instance_name, b_scores, r_scores):
    os.makedirs(f"{dir}/{instance_name}", exist_ok=True)
    
    diff = np.abs(b_scores - r_scores)

    plt_fig = Figure(figsize=(6,6))
    plt_axis = plt_fig.subplots()
    plt_axis.xaxis.set_ticks_position("none")
    plt_axis.yaxis.set_ticks_position("none")
    plt_axis.set_xticklabels([])
    plt_axis.set_yticklabels([])
    plt_axis.grid(visible=False)

    cmap = LinearSegmentedColormap.from_list("Ex", ["white", "orange", "red", "magenta"])
    cmap.set_bad(color="black")
    vmin, vmax = 0, 2
    heat_map = plt_axis.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax)

    axis_separator = make_axes_locatable(plt_axis)
    colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
    plt_fig.colorbar(heat_map, orientation="horizontal", cax=colorbar_axis)

    plt_fig.savefig(f"{dir}/{instance_name}/{instance_name}_exp_comparison.png")