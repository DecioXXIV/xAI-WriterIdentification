import os, torch, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix

from model import NN_Classifier
from utils import Confidence_Test_DataLoader, load_rgb_mean_std

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def get_args():
    parser = ArgumentParser()
    parser.add_argument("-base_id", type=str, required=True)
    parser.add_argument("-ret_id", type=str, required=True)
    parser.add_argument("-subject", type=str, required=True, choices=["confidence", "explanations"])
    parser.add_argument("-mode", type=str, choices=["scan", "random"])
    parser.add_argument("-mult_factor", type=int, default=1)
    parser.add_argument("-iters", type=int, default=3)
    
    return parser.parse_args()

def validate_args(args) -> bool:
    BASELINE, RETRAINED = args.base_id, args.ret_id
    RET_BASE, _ = RETRAINED.split(':')
    
    if RET_BASE != BASELINE: return False
    else: return True

def load_metadata(metadata_path) -> dict:
    try:
        with open(metadata_path, 'r') as jf: return json.load(jf)
    except json.JSONDecodeError as e: raise ValueError(f"Error occurred while decoding JSON file '{metadata_path}': {e}")
    except Exception as e: raise FileNotFoundError(f"Error occurred while reading metadata file '{metadata_path}': {e}")

def load_model(cp, cp_ft, num_classes):
    model = NN_Classifier(num_classes=num_classes, mode="frozen", cp_path=cp)
    model.load_state_dict(torch.load(cp_ft)["model_state_dict"])
    model.eval()
    
    return model

def get_instances(root_dir, dataset, classes):
    os.makedirs(f"{root_dir}/test_instances", exist_ok=True)
    dataset_dir = f"./../../datasets/{dataset}"
    
    for c in classes:
        os.makedirs(f"{root_dir}/test_instances/{c}")
        
        test_instances = list()
        if dataset == "CEDAR-Letter": test_instances = [f for f in os.listdir(f"{dataset_dir}/processed/{c}") if "c" in f]
        if dataset == "CVL": test_instances = [f for f in os.listdir(f"{dataset_dir}/processed/{c}") if ("-3" in f or "-7" in f)]
        
        for f in test_instances:
            source = f"{dataset_dir}/{c}/{f}"
            dest = f"{root_dir}/test_instances/{c}/{f}"
            
            os.system(f"cp {source} {dest}")

def models_inference(models, dataset, instances, device, root_dir, classes, mode, mult_factor, iter):
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
            for c_idx in enumerate(classes): values.append(out_b[i][c_idx].item())
            for c_idx in enumerate(classes): values.append(out_r[i][c_idx].item())
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
        prefix = instance[0:4]
        label = int(prefix)
        
        stats = df_probs[(df_probs["Instance"] == instance)]
        
        baseline_mean = np.mean(stats[f"P{label}_Baseline"].tolist())
        baseline_var = np.var(stats[f"P{label}_Baseline"].tolist())
        ret_mean = np.mean(stats[f"P{label}_Ret"].tolist())
        ret_var = np.var(stats[f"P{label}_Ret"].tolist())
        
        delta_mean = baseline_mean - ret_mean
        
        df_stats.loc[len(df_stats)] = [instance, baseline_mean, baseline_var, ret_mean, ret_var, delta_mean]
    
    if mode == "scan": df_stats.to_csv(f"{root_dir}/fact_{mult_factor}/comparison_stats.csv", index=False, header=True)
    if mode == "random": df_stats.to_csv(f"{root_dir}/iter_{iter}/comparison_stats.csv", index=False, header=True)
    
def produce_comparison_reports(idx_to_c, labels, preds_b, preds_r, instances, root_dir, mode, mult_factor, iter):
    label_class_names = [idx_to_c[id_] for id_ in labels]
    pred_b_class_names = [idx_to_c[id_] for id_ in preds_b]
    pred_r_class_names = [idx_to_c[id_] for id_ in preds_r]
    
    df_preds = pd.DataFrame(columns=["Instance", "True_Label", "Pred_B_Label", "Pred_R_Label"])
    
    df_preds["Instance"] = instances
    df_preds["True_Label"] = label_class_names
    df_preds["Pred_B_Label"] = pred_b_class_names
    df_preds["Pred_R_Label"] = pred_r_class_names
    
    if mode == "scan": df_preds.to_csv(f"{root_dir}/fact_{mult_factor}/comparison_labels.csv", index=False, header=True)
    if mode == "random": df_preds.to_csv(f"{root_dir}/iter_{iter}/comparison_labels.csv", index=False, header=True)
    
    produce_confusion_matrix("baseline")
    produce_confusion_matrix("retrained")

def produce_confusion_matrix(exp_type, label_class_names, pred_class_names, target_names, mode, mult_factor, iter):
    cm = None
    if exp_type == "baseline": cm = confusion_matrix()
    if exp_type == "retrained": cm = confusion_matrix()
    
    # CONTINUARE! ###

def execute_pair_test(model_b, model_r, dl, instances, device, root_dir, classes, mode, mult_factor=None, iter=None):
    dataset_, set_ = dl.load_data()
    target_names = list(dataset_.class_to_idx.keys())
    c_to_idx = dataset_.class_to_idx
    idx_to_c = {c_to_idx[k]: k for k in list(c_to_idx.keys())}
    
    ### MODELS INFERENCE ###
    labels, preds_b, preds_r = models_inference((model_b, model_r), set_, instances, device, root_dir, classes, mode, mult_factor, iter)
    ### ################ ###
    
    ### DELTA REPORTS ###
    produce_delta_reports(instances, root_dir, mode, mult_factor, iter)
    ### ############# ###
    
    ### COMPARISON REPORTS ###
    produce_comparison_reports(idx_to_c, labels, preds_b, preds_r, instances, root_dir, mode, mult_factor, iter)
    ### ################## ###

def pair_confidence_test(baseline, retrained, mode, mult_factor, iters):
    CWD = os.getcwd()
    BASELINE_METADATA_PATH = CWD + f"/../../log/{baseline}-metadata.json"
    RETRAINED_METADATA_PATH = CWD + f"/../../log/{retrained}-metadata.json"
    
    BASELINE_METADATA = load_metadata(BASELINE_METADATA_PATH)
    RETRAINED_METADATA = load_metadata(RETRAINED_METADATA_PATH)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CROP_SIZE = 380
    
    cp = CWD + "/../cp/Test_3_TL_val_best_model.pth"
    cp_baseline = CWD + f"/tests/{BASELINE}/output/checkpoints/Test_{BASELINE}_MLC_val_best_model.pth"
    cp_ret = CWD + f"/tests/{RETRAINED}/output/checkpoints/Test_{RETRAINED}_MLC_val_best_model.pth"
    
    BASELINE_CLASSES, RETRAINED_CLASSES = BASELINE_METADATA["CLASSES"], RETRAINED_METADATA["CLASSES"]
    model_baseline, model_retrained = load_model(cp, cp_baseline, len(BASELINE_CLASSES)), load_model(cp, cp_ret, len(RETRAINED_CLASSES))
    
    model_baseline = model_baseline.to(DEVICE)
    model_retrained = model_retrained.to(DEVICE)
    
    root_dir = f"./pairs/{baseline}/VERSUS-{retrained}/confidence/{mode}"
    os.makedirs(root_dir, exist_ok=True)
    
    DATASET = BASELINE_METADATA["DATASET"]
    CLASSES = list(BASELINE_METADATA["CLASSES"].keys())
    get_instances(root_dir, DATASET, CLASSES)
    
    mean_, std_ = load_rgb_mean_std(f"{CWD}/tests/{BASELINE}")
    
    dl = Confidence_Test_DataLoader(f"{root_dir}/test_instances", CLASSES, 1, CROP_SIZE, False, mode, mult_factor, mean_, std_, False)
    instances = list()
    for c in CLASSES: instances.extend(os.listdir(f"{root_dir}/test_instances/{c}"))
    
    if mode == "scan": execute_pair_test(model_baseline, model_retrained, dl, instances, DEVICE, root_dir, CLASSES, mode, mult_factor, None)
    if mode == "random": 
        for iter in (0, iters): execute_pair_test(model_baseline, model_retrained, dl, instances, DEVICE, root_dir, CLASSES, mode, None, iter+1)
    
    
    
    
    
    

### #### ###
### MAIN ###
### #### ###
if __name__ == '__main__':
    args = get_args()
    
    if not validate_args(args):
        print("*** PROBLEMS! ***")
        exit(1)
    
    BASELINE, RETRAINED = args.base_id, args.ret_id
    SUBJECT = args.subject
    MODE = args.mode
    MULT_FACTOR, ITERS = args.mult_factor, args.iters
    
    match MODE:
        case 'scan': print(f"*** BEGINNING OF PAIR TEST -> '{MODE}' MODE, MULT_FACTOR = {MULT_FACTOR}***")
        case 'random': print(f"*** BEGINNING OF PAIR TEST -> '{MODE}' MODE, ITERS = {ITERS}***")
    
    print(f"Baseline Experiment: {BASELINE}")
    print(f"Re-Trained Experiment: {RETRAINED}")
    
    
    pair_confidence_test(BASELINE, RETRAINED, MODE, MULT_FACTOR, ITERS)
    
    # execute_pair_explanation_test(BASELINE, RETRAINED) -> TO DO!