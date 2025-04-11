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
from classifiers.utils.dataloader_utils import Eval_Test_DataLoader

from xai.utils.explanations_utils import reduce_scores, assign_attr_scores_to_mask

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
CLASSIFIERS_ROOT = "./classifiers"
XAI_ROOT = "./xai"
EVALS_ROOT = "./evals"

def retrieve_test_instances(root_dir, dataset, classes):
    os.makedirs(f"{root_dir}/test_instances", exist_ok=True)
    dataset_dir = f"{DATASET_ROOT}/{dataset}"
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
def pair_confidence_test(baseline_id, retrained_id, logger):
    BASELINE_METADATA = load_metadata(f"{LOG_ROOT}/{baseline_id}-metadata.json", logger)
    RETRAINED_METADATA = load_metadata(f"{LOG_ROOT}/{retrained_id}-metadata.json", logger)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATASET = BASELINE_METADATA["DATASET"]
    CROP_SIZE = BASELINE_METADATA["FINE_TUNING_HP"]["crop_size"]
    B_MODEL_TYPE, R_MODEL_TYPE = BASELINE_METADATA["MODEL_TYPE"], RETRAINED_METADATA["MODEL_TYPE"]
    BASELINE_CLASSES, RETRAINED_CLASSES = BASELINE_METADATA["CLASSES"], RETRAINED_METADATA["CLASSES"]
    cp_base = get_model_base_checkpoint(B_MODEL_TYPE)
    
    model_b, _ = load_model(B_MODEL_TYPE, len(BASELINE_CLASSES), "frozen", cp_base, "test", baseline_id, BASELINE_METADATA, DEVICE, logger)
    model_r, _ = load_model(R_MODEL_TYPE, len(RETRAINED_CLASSES), "frozen", cp_base, "test", retrained_id, RETRAINED_METADATA, DEVICE, logger)
    
    root_dir = f"{EVALS_ROOT}/bvr_comparisons/{baseline_id}/VERSUS-{retrained_id}/confidence"
    os.makedirs(root_dir, exist_ok=True)
    
    CLASSES = list(BASELINE_METADATA["CLASSES"].keys())

    mean_b, std_b = BASELINE_METADATA["FINE_TUNING_HP"]["mean"], BASELINE_METADATA["FINE_TUNING_HP"]["std"]
    mean_r, std_r = RETRAINED_METADATA["FINE_TUNING_HP"]["mean"], RETRAINED_METADATA["FINE_TUNING_HP"]["std"]
    
    retrieve_test_instances(root_dir, DATASET, CLASSES)
    dl_b = Eval_Test_DataLoader(f"{root_dir}/test_instances", CLASSES, 1, CROP_SIZE, 2, mean_b, std_b)
    dl_r = Eval_Test_DataLoader(f"{root_dir}/test_instances", CLASSES, 1, CROP_SIZE, 2, mean_r, std_r)
    
    pages = list()
    for c in CLASSES: pages.extend(os.listdir(f"{root_dir}/test_instances/{c}"))
    
    execute_pair_confidence_test(model_b, model_r, dl_b, dl_r, pages, DEVICE, root_dir, CLASSES)
    os.system(f"rm -r {root_dir}/test_instances")
    
def pair_explanations_test(baseline_id, retrained_id, xai_algorithm, xai_mode, logger):
    BASELINE_METADATA = load_metadata(f"{LOG_ROOT}/{baseline_id}-metadata.json", logger)
    RETRAINED_METADATA = load_metadata(f"{LOG_ROOT}/{retrained_id}-metadata.json", logger)
    
    DATASET = BASELINE_METADATA["DATASET"]

    B_EXP_DIR = BASELINE_METADATA[f"{xai_algorithm}_{xai_mode}_METADATA"]["DIR_NAME"]
    R_EXP_DIR = RETRAINED_METADATA[f"{xai_algorithm}_{xai_mode}_METADATA"]["DIR_NAME"]

    PATCH_W = BASELINE_METADATA[f"{xai_algorithm}_{xai_mode}_METADATA"]["PATCH_DIM"]["WIDTH"]
    PATCH_H = BASELINE_METADATA[f"{xai_algorithm}_{xai_mode}_METADATA"]["PATCH_DIM"]["HEIGHT"]

    root_dir = f"{EVALS_ROOT}/bvr_comparisons/{baseline_id}/VERSUS-{retrained_id}/explanations/{xai_algorithm}"
    os.makedirs(root_dir, exist_ok=True)

    test_instance_names = os.listdir(f"{XAI_ROOT}/explanations/patches_{PATCH_W}x{PATCH_H}_removal/{R_EXP_DIR}")
    test_instance_names.remove("rgb_train_stats.pkl")
    test_instance_names.remove(f"{retrained_id}-xai_metadata.json")

    for name in test_instance_names:
        base_scores_path = f"{XAI_ROOT}/explanations/patches_{PATCH_W}x{PATCH_H}_removal/{B_EXP_DIR}/{name}/{name}_scores.pkl"
        ret_scores_path = f"{XAI_ROOT}/explanations/patches_{PATCH_W}x{PATCH_H}_removal/{R_EXP_DIR}/{name}/{name}_scores.pkl"

        base_scores, ret_scores = None, None
        with open(base_scores_path, "rb") as f: base_scores = pickle.load(f)
        with open(ret_scores_path, "rb") as f: ret_scores = pickle.load(f)

        mask = Image.open(f"{XAI_ROOT}/masks/{DATASET}_mask_{PATCH_W}x{PATCH_H}.png")

        base_scores_reduced, ret_scores_reduced = reduce_scores(mask, base_scores), reduce_scores(mask, ret_scores)
        b_mask_with_scores, r_mask_with_scores  = assign_attr_scores_to_mask(mask, base_scores_reduced), assign_attr_scores_to_mask(mask, ret_scores_reduced)

        produce_explanation_comparison(root_dir, name, b_mask_with_scores, r_mask_with_scores)
        logger.info(f"Processed Test Instance: {name}")

### ############################## ###
### PAIR CONFIDENCE TEST FUNCTIONS ###
### ############################## ###
def execute_pair_confidence_test(model_b, model_r, dl_b, dl_r, page_names, device, root_dir, classes):
    model_inference("Baseline", model_b, dl_b, page_names, device, root_dir, classes)
    model_inference("Re-Trained", model_r, dl_r, page_names, device, root_dir, classes)
    produce_comparison_report(page_names, root_dir)

def model_inference(exp_type, model, dl, pages, device, root_dir, classes):
    torch.backends.cudnn.benchmark = True
    dataset_, set_ = dl.load_data()
    c_to_idx = dataset_.class_to_idx
    idx_to_c = {c_to_idx[k]: k for k in list(c_to_idx.keys())}
    
    columns = ["Page", "Crop_No"]
    for c in classes: columns.append(f"P{c}_{exp_type}")
    columns.extend(["True_Label", f"Pred_Label_{exp_type}"])
    
    df_probs = pd.DataFrame(columns=columns)
    j = 0
    
    for data, target in tqdm(set_, desc=f"{exp_type} Model Inference"):
        data = data.to(device)
        _, ncrops, c, h, w = data.size()
        
        with torch.no_grad():
            out = model(data.view(-1, c, h, w))
            out = F.softmax(out, dim=1)
            max_indices = out.max(dim=1)[1].cpu().numpy()
            
        for i in range(0, ncrops):
            values = [pages[j], i+1]
            for _ in range(0, len(idx_to_c)): values.append(None)
            values.extend([target.item(), max_indices[i]])
            df_probs.loc[len(df_probs)] = values
            
            for c_idx in idx_to_c.keys(): 
                c_name = idx_to_c[c_idx]
                df_probs.at[len(df_probs)-1, f"P{c_name}_{exp_type}"] = out[i][c_idx].item()
        
        j += 1
    
    df_probs.to_csv(f"{root_dir}/{exp_type}_crop_probs.csv", index=False, header=True)
            
def produce_comparison_report(page_names, root_dir):
    df_stats = pd.DataFrame(columns=["Instance", "Baseline_Mean", "Baseline_Var", "Ret_Mean", "Ret_Var", "Delta_Mean"])
    df_probs_b = pd.read_csv(f"{root_dir}/Baseline_crop_probs.csv", header=0)
    df_probs_r = pd.read_csv(f"{root_dir}/Re-Trained_crop_probs.csv", header=0)
    
    for p_name in page_names:
        label = int(p_name[0:4])
        stats_b = df_probs_b[(df_probs_b["Page"]) == p_name]
        stats_r = df_probs_r[(df_probs_r["Page"]) == p_name]
        
        b_mean = np.mean(stats_b[f"P{label}_Baseline"].tolist())
        b_var = np.var(stats_b[f"P{label}_Baseline"].tolist())
        ret_mean = np.mean(stats_r[f"P{label}_Re-Trained"].tolist())
        ret_var = np.var(stats_r[f"P{label}_Re-Trained"].tolist())
        
        delta_mean = ret_mean - b_mean
        
        df_stats.loc[len(df_stats)] = [p_name, b_mean, b_var, ret_mean, ret_var, delta_mean]
    
    df_stats.to_csv(f"{root_dir}/comparison_stats.csv", index=False, header=True)

### ################################ ###
### PAIR EXPLANATIONS TEST FUNCTIONS ###
### ################################ ###
def produce_explanation_comparison(root_dir, instance_name, b_scores, r_scores):
    os.makedirs(f"{root_dir}/{instance_name}", exist_ok=True)
    
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

    plt_fig.savefig(f"{root_dir}/{instance_name}/{instance_name}_exp_comparison.png")