import os, torch, json, itertools, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from classifiers.classifier_NN.model import NN_Classifier
from classifiers.classifier_NN.utils import Confidence_Test_DataLoader, load_rgb_mean_std

from xai.explainers import reduce_scores, assign_attr_scores_to_mask

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
XAI_ROOT = "./xai"
PAIR_TEST_ROOT = "./bvr_comparisons"

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
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase", "GLimeBinomial"])
    
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
    dataset_dir = f"./datasets/{dataset}"
    
    for c in classes:
        os.makedirs(f"{root_dir}/test_instances/{c}", exist_ok=True)
        
        test_instances = list()
        if dataset == "CEDAR_Letter": test_instances = [f for f in os.listdir(f"{dataset_dir}/processed/{c}") if "c" in f]
        if dataset == "CVL": test_instances = [f for f in os.listdir(f"{dataset_dir}/processed/{c}") if ("-3" in f or "-7" in f)]
        
        for f in test_instances:
            source = f"{dataset_dir}/processed/{c}/{f}"
            dest = f"{root_dir}/test_instances/{c}/{f}"
            
            os.system(f"cp {source} {dest}")

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
    produce_confusion_matrix(dir, "baseline", label_class_names, pred_b_class_names, target_names)
    produce_confusion_matrix(dir, "retrained", label_class_names, pred_r_class_names, target_names)

def produce_confusion_matrix(dir, test_type, label_class_names, pred_class_names, target_names):
    cm = confusion_matrix(label_class_names, pred_class_names, labels=target_names)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted label\naccuracy = {:0.4f}; misclass = {:0.4f}'.format(accuracy, misclass))

    plt.savefig(f"{dir}/{test_type}_confusion_matrix.png")

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
    
    ### TODO! New Metrics to Compare the Explanations

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

### ################### ###
### PRINCIPAL FUNCTIONS ###
### ################### ###
def pair_confidence_test(baseline, retrained, mode, mult_factor, iters):
    BASELINE_METADATA_PATH = f"{LOG_ROOT}/{baseline}-metadata.json"
    RETRAINED_METADATA_PATH =  f"{LOG_ROOT}/{retrained}-metadata.json"
    
    BASELINE_METADATA = load_metadata(BASELINE_METADATA_PATH)
    RETRAINED_METADATA = load_metadata(RETRAINED_METADATA_PATH)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CROP_SIZE = 380

    B_MODEL_TYPE, R_MODEL_TYPE = BASELINE_METADATA["MODEL_TYPE"], RETRAINED_METADATA["MODEL_TYPE"]
    
    CLASSIFIER_ROOT = f"./classifiers/classifier_{B_MODEL_TYPE}"
    cp = f"{CLASSIFIER_ROOT}/../cp/Test_3_TL_val_best_model.pth"
    cp_baseline = f"{CLASSIFIER_ROOT}/tests/{BASELINE}/output/checkpoints/Test_{BASELINE}_MLC_val_best_model.pth"
    cp_ret = f"{CLASSIFIER_ROOT}/tests/{RETRAINED}/output/checkpoints/Test_{RETRAINED}_MLC_val_best_model.pth"
    
    BASELINE_CLASSES, RETRAINED_CLASSES = BASELINE_METADATA["CLASSES"], RETRAINED_METADATA["CLASSES"]
    model_baseline, model_retrained = load_model(cp, cp_baseline, len(BASELINE_CLASSES)), load_model(cp, cp_ret, len(RETRAINED_CLASSES))
    
    model_baseline = model_baseline.to(DEVICE)
    model_retrained = model_retrained.to(DEVICE)
    
    root_dir = f"{PAIR_TEST_ROOT}/tests/{baseline}/VERSUS-{retrained}/confidence/{mode}"
    os.makedirs(root_dir, exist_ok=True)
    
    DATASET = BASELINE_METADATA["DATASET"]
    CLASSES = list(BASELINE_METADATA["CLASSES"].keys())
    get_instances(root_dir, DATASET, CLASSES)
    
    mean_, std_ = load_rgb_mean_std(f"{CLASSIFIER_ROOT}/tests/{BASELINE}")
    
    dl = Confidence_Test_DataLoader(f"{root_dir}/test_instances", CLASSES, 1, CROP_SIZE, False, mode, mult_factor, mean_, std_, False)
    instances = list()
    for c in CLASSES: instances.extend(os.listdir(f"{root_dir}/test_instances/{c}"))
    
    if mode == "scan": execute_pair_confidence_test(model_baseline, model_retrained, dl, instances, DEVICE, root_dir, CLASSES, mode, mult_factor, None)
    if mode == "random": 
        for iter in range(0, iters): execute_pair_confidence_test(model_baseline, model_retrained, dl, instances, DEVICE, root_dir, CLASSES, mode, None, iter+1)
    
    os.system(f"rm -r {root_dir}/test_instances")

def pair_explanations_test(baseline, retrained, xai_algorithm):
    BASELINE_METADATA_PATH = f"{LOG_ROOT}/{baseline}-metadata.json"
    RETRAINED_METADATA_PATH =  f"{LOG_ROOT}/{retrained}-metadata.json"
    
    BASELINE_METADATA = load_metadata(BASELINE_METADATA_PATH)
    RETRAINED_METADATA = load_metadata(RETRAINED_METADATA_PATH)

    B_EXP_DIR = BASELINE_METADATA[f"{xai_algorithm}_METADATA"]["DIR_NAME"]
    R_EXP_DIR = RETRAINED_METADATA[f"{xai_algorithm}_METADATA"]["DIR_NAME"]

    BLOCK_W = BASELINE_METADATA[f"{xai_algorithm}_METADATA"]["BLOCK_DIM"]["WIDTH"]
    BLOCK_H = BASELINE_METADATA[f"{xai_algorithm}_METADATA"]["BLOCK_DIM"]["HEIGHT"]

    root_dir = f"{PAIR_TEST_ROOT}/tests/{baseline}/VERSUS-{retrained}/explanations/{xai_algorithm}"
    os.makedirs(root_dir, exist_ok=True)

    test_instances = os.listdir(f"{XAI_ROOT}/explanations/patches_{BLOCK_W}x{BLOCK_H}_removal/{R_EXP_DIR}")
    test_instances.remove("rgb_train_stats.pkl")

    for i in test_instances:
        base_scores_path = f"{XAI_ROOT}/explanations/patches_{BLOCK_W}x{BLOCK_H}_removal/{B_EXP_DIR}/{i}/{i}_scores.pkl"
        ret_scores_path = f"{XAI_ROOT}/explanations/patches_{BLOCK_W}x{BLOCK_H}_removal/{R_EXP_DIR}/{i}/{i}_scores.pkl"

        base_scores, ret_scores = None, None
        with open(base_scores_path, "rb") as f: base_scores = pickle.load(f)
        with open(ret_scores_path, "rb") as f: ret_scores = pickle.load(f)

        mask = Image.open(f"{XAI_ROOT}/def_mask.png")

        base_scores_reduced = reduce_scores(mask, base_scores)
        ret_scores_reduced = reduce_scores(mask, ret_scores)

        b_mask_with_scores = assign_attr_scores_to_mask(mask, base_scores_reduced)
        r_mask_with_scores = assign_attr_scores_to_mask(mask, ret_scores_reduced)

        produce_explanation_comparison(root_dir, i, b_mask_with_scores, r_mask_with_scores)
        print(f"Processed Test Instance: {i}")

### #### ###
### MAIN ###
### #### ###
if __name__ == '__main__':
    args = get_args()
    
    if not validate_args(args):
        print("*** THERE ARE INCONGRUENCES WITHIN THE TEST TO BE COMPARED! ***")
        exit(1)
    
    BASELINE, RETRAINED = args.base_id, args.ret_id
    SUBJECT = args.subject
    MODE = args.mode
    MULT_FACTOR, ITERS = args.mult_factor, args.iters
    XAI_ALGORITHM = args.xai_algorithm
    
    print(f"Baseline Experiment: {BASELINE}")
    print(f"Re-Trained Experiment: {RETRAINED}")
    
    if SUBJECT == "confidence":
        if MODE == "scan": print(f"*** BEGINNING OF CONFIDENCE PAIR TEST -> '{MODE}' MODE, MULT_FACTOR = {MULT_FACTOR} ***")
        if MODE == "random": print(f"*** BEGINNING OF CONFIDENCE PAIR TEST -> '{MODE}' MODE, ITERS = {ITERS} ***")
        pair_confidence_test(BASELINE, RETRAINED, MODE, MULT_FACTOR, ITERS)
    if SUBJECT == "explanations":
        print(f"*** BEGINNING OF EXPLANATIONS PAIR TEST -> '{XAI_ALGORITHM}' ALGORITHM") 
        pair_explanations_test(BASELINE, RETRAINED, XAI_ALGORITHM)

    torch.cuda.empty_cache()