import os, torch, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import NN_Classifier
from utils import Confidence_Test_DataLoader, load_rgb_mean_std
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from argparse import ArgumentParser

### ################### ###
### PRINCIPAL FUNCTIONS ###
### ################### ###
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--mult_factor", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    
    return parser.parse_args()

def execute_comparison_test(ret, instances, dataset, models, target_names, idx_to_c, iteration=None):
    if iteration is not None:
        os.makedirs(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}/{iteration}", exist_ok=True)
    else:
        os.makedirs(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}/fact{MULT_FACTOR}", exist_ok=True)
    
    labels, preds_baseline, preds_ret = model_inference(ret, dataset, models, iteration)
    produce_delta_reports(ret, instances, iteration)
    produce_comparison_reports(ret, instances, labels, preds_baseline, preds_ret, target_names, idx_to_c, iteration)

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def model_inference(ret, dataset, models, iteration):
    if iteration is not None:
        print(f"Beginning of Model Inference -> Iteration = {iteration}")
    else:
        print("Beginning of Model Inference")
    
    labels, preds_baseline, preds_ret = list(), list(), list()
    df_probs = pd.DataFrame(columns=["Instance", "Crop_No", "P1_Baseline", "P2_Baseline", "P3_Baseline", "P1_Ret", "P2_Ret", "P3_Ret", "True Label", "Pred Baseline", "Pred Ret"])
    j = 0
    
    for data, target in tqdm(dataset):
        labels += list(target.numpy())
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        bs, ncrops, c, h, w = data.size()
        
        model_baseline, model_ret = models
        
        with torch.no_grad():
            # Models (Baseline and Re-Trained) Inference on Instances
            out_baseline, out_ret = model_baseline(data.view(-1, c, h, w)), model_ret(data.view(-1, c, h, w))
            out_baseline, out_ret = F.softmax(out_baseline, dim=1), F.softmax(out_ret, dim=1)
            # out_baseline, out_ret = F.sigmoid(out_baseline), F.sigmoid(out_ret)
            
            # For each instance we take into account all the crops
            # Each crops is attributed by following the max value of the model's output
            max_index_baseline = out_baseline.max(dim=1)[1].cpu().detach().numpy()
            max_index_ret = out_ret.max(dim=1)[1].cpu().detach().numpy()
            
            # Each instance is attributed by following the Majority Voting Rule on its crops
            final_max_index_baseline, final_max_index_ret = list(), list()
            
            max_index_over_crops_baseline = max_index_baseline.reshape(bs, ncrops)
            max_index_over_crops_ret = max_index_ret.reshape(bs, ncrops)
            
            for s in range(0, bs):
                final_max_index_baseline.append(np.argmax(np.bincount(max_index_over_crops_baseline[s, :])))
                final_max_index_ret.append(np.argmax(np.bincount(max_index_over_crops_ret[s, :])))
            
            preds_baseline += list(final_max_index_baseline)
            preds_ret += list(final_max_index_ret)
        
        for i in range(0, ncrops):
            df_probs.loc[len(df_probs)] = [instances[j], i,
                                           out_baseline[i][0].item(), out_baseline[i][1].item(), out_baseline[i][2].item(),
                                           out_ret[i][0].item(), out_ret[i][1].item(), out_ret[i][2].item(),
                                           target.cpu().numpy()[0], max_index_baseline[i], max_index_ret[i]]
        
        j += 1
    
    if iteration is not None:
        df_probs.to_csv(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}/{iteration}/comparison_probs_CEDAR-Letter-0005.csv", index=False, header=True)
    else:
        df_probs.to_csv(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}/fact{MULT_FACTOR}/comparison_probs_CEDAR-Letter-0005.csv", index=False, header=True)
    
    return labels, preds_baseline, preds_ret 

def produce_delta_reports(ret, instances, iteration):
    if iteration is not None:
        print(f"Producing Delta Reports -> Iteration = {iteration}")
        df_probs = pd.read_csv(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}/{iteration}/comparison_probs_CEDAR-Letter-0005.csv", header=0)
    else:
        print("Producing Delta Reports")
        df_probs = pd.read_csv(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}/fact{MULT_FACTOR}/comparison_probs_CEDAR-Letter-0005.csv", header=0)
        
    df = pd.DataFrame(columns=["Instance", "Baseline_Mean", "Baseline_Var", "Ret_Mean", "Ret_Var", "Delta_Mean"])
    
    for instance in instances:
        label, prefix = None, instance[0:4]
        match prefix:
            case '0001': label = 1
            case '0002': label = 2
            case '0003': label = 3
        
        stats = df_probs[(df_probs["Instance"] == instance)]
        
        baseline_mean = np.mean(stats[f"P{label}_Baseline"].tolist())
        baseline_var = np.var(stats[f"P{label}_Baseline"].tolist())
        ret_mean = np.mean(stats[f"P{label}_Ret"].tolist())
        ret_var = np.var(stats[f"P{label}_Ret"].tolist())
        delta_mean = baseline_mean - ret_mean
        
        df.loc[len(df)] = [instance, baseline_mean, baseline_var, ret_mean, ret_var, delta_mean]
    
    if iteration is not None:
        df.to_csv(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}/{iteration}/comparison_stats_CEDAR-Letter-0005.csv", index=False)
    else:
        df.to_csv(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}/fact{MULT_FACTOR}/comparison_stats_CEDAR-Letter-0005.csv", index=False)
            
def produce_comparison_reports(ret, instances, labels, preds_baseline, preds_ret, target_names, idx_to_c, iteration):
    if iteration is not None:
        print(f"Producing Comparison Reports -> Iteration = {iteration}")
    else:
        print("Producing Comparison Reports")
    
    label_class_names = [idx_to_c[id_] for id_ in labels]
    pred_baseline_class_names = [idx_to_c[id_] for id_ in preds_baseline]
    pred_ret_class_names = [idx_to_c[id_] for id_ in preds_ret]
    
    df_preds = pd.DataFrame(columns=["Instance", "True Label", "Predicted Label Baseline", "Predicted Label Retrained"])
    
    df_preds["Instance"] = instances
    df_preds["True Label"] = label_class_names
    df_preds["Predicted Label Baseline"] = pred_baseline_class_names 
    df_preds["Predicted Label Retrained"] = pred_ret_class_names
    
    if iteration is not None:
        df_preds.to_csv(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}/{iteration}/comparison_labels_CEDAR-Letter-0005.csv", index=False, header=True)
    else: 
        df_preds.to_csv(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}/fact{MULT_FACTOR}/comparison_labels_CEDAR-Letter-0005.csv", index=False, header=True)
    
    produce_confusion_matrix(ret, label_class_names, pred_baseline_class_names, target_names, 'baseline', iteration)
    produce_confusion_matrix(ret, label_class_names, pred_ret_class_names, target_names, 're-trained', iteration)

def produce_confusion_matrix(ret, label_class_names, pred_class_names, target_names, model_type, iteration):
    if iteration is not None:
        print(f"Producing Confusion Matrix -> {model_type} Model, Iteration = {iteration}")
    else:
        print(f"Producing Confusion Matrix -> {model_type} Model")
    
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
    
    assert model_type in ['baseline', 're-trained']
    if iteration is not None:
        plt.savefig(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}/{iteration}/CEDAR-Letter-0005_{model_type}_confusion_matrix_test.png")
    else:
        plt.savefig(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}/fact{MULT_FACTOR}/CEDAR-Letter-0005_{model_type}_confusion_matrix_test_fact.png")

### #### ###
### MAIN ###
### #### ###
if __name__ == '__main__':
    args = get_args()
    
    MODE = args.mode
    MULT_FACTOR = args.mult_factor
    ITERS = args.iters
    CWD = os.getcwd()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CROP_SIZE = 380
    
    match MODE:
        case 'scan': print(f"Beginning of Confidence Pair Test -> SCAN mode, mult_factor = {MULT_FACTOR}")
        case 'randoms': print(f"Beginning of Confidence Pair Test -> RANDOMS mode, iters = {ITERS}")
        
    DATASET_DIR_BASELINE = CWD + f"/tests/CEDAR-Letter-0005"
    OUTPUT_DIR_BASELINE = CWD + f"/tests/CEDAR-Letter-0005/output"
    cp_base = f"./cp/Test_3_TL_val_best_model.pth"
    
    RETRAINS = list()
    RETRAINS.append("CEDAR-Letter-0005_ret0.05_all-3")
    RETRAINS.append("CEDAR-Letter-0005_ret0.05_c1-2")
    RETRAINS.append("CEDAR-Letter-0005_ret0.05_c2-3")
    RETRAINS.append("CEDAR-Letter-0005_ret0.05_c3-1")
    
    ### BASELINE ###
    cp_baseline = f"{OUTPUT_DIR_BASELINE}/checkpoints/Test_CEDAR-Letter-0005_MLC_val_best_model.pth"
    mean_b, std_b = load_rgb_mean_std(f"{DATASET_DIR_BASELINE}/train")

    model_baseline = NN_Classifier(num_classes=3, mode='frozen', cp_path=cp_base)
    model_baseline = model_baseline.to(DEVICE)
    model_baseline.load_state_dict(torch.load(cp_baseline)['model_state_dict'])
    model_baseline.eval()
    
    os.makedirs("./comparison_tests", exist_ok=True)
    os.makedirs("./comparison_tests/CEDAR-Letter-0005", exist_ok=True)
    
    ### RE-TRAINS ###
    for ret in RETRAINS:
        print(f"Baseline VS Re-Train: {ret}")
        DATASET_DIR_RET = CWD + f"/tests/{ret}"
        OUTPUT_DIR_RET = CWD + f"/tests/{ret}/output"
        
        cp_ret = f"{OUTPUT_DIR_RET}/checkpoints/Test_{ret}_MLC_val_best_model.pth"
        mean_r, std_r = load_rgb_mean_std(f"{DATASET_DIR_RET}/train")

        model_ret = NN_Classifier(num_classes=3, mode='frozen', cp_path=cp_base)
        model_ret = model_ret.to(DEVICE)
        model_ret.load_state_dict(torch.load(cp_ret)['model_state_dict'])
        model_ret.eval()

        # DATASET LOADING
        instances = list()
        instances.extend(os.listdir(f"{DATASET_DIR_BASELINE}/test/1"))
        instances.extend(os.listdir(f"{DATASET_DIR_BASELINE}/test/2"))
        instances.extend(os.listdir(f"{DATASET_DIR_BASELINE}/test/3"))

        dl = Confidence_Test_DataLoader(f"{DATASET_DIR_BASELINE}/test", 1, CROP_SIZE, False, MODE, MULT_FACTOR, mean_r, std_r, False)
        dataset = dl.generate_dataset()
        _, set_ = dl.load_data()

        target_names = list(dataset.class_to_idx.keys())
        c_to_idx = dataset.class_to_idx
        idx_to_c = {c_to_idx[k]: k for k in list(c_to_idx.keys())}
        
        os.makedirs(f"./comparison_tests/CEDAR-Letter-0005/baseline_vs_{ret}_mode_{MODE}", exist_ok=True)
        
        match MODE:
            case 'scan': execute_comparison_test(ret, instances, set_, (model_baseline, model_ret), target_names, idx_to_c)
            case 'randoms':
                for it in range(0, ITERS): execute_comparison_test(ret, instances, set_, (model_baseline, model_ret), target_names, idx_to_c, it+1)

        print("Comparison Test Completed!\n")
        torch.cuda.empty_cache()