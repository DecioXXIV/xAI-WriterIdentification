import os, json
import multiprocessing as mp
import numpy as np
import pandas as pd
from itertools import combinations

DATASETS_ROOT = "./datasets"
XAI_ROOT = "./xai"
EVAL_ROOT = "./evals"

def get_instances(dataset, classes):
    instances = list()
    for c in classes:
        c_instances = os.listdir(f"{DATASETS_ROOT}/{dataset}/processed/{c}")
        for inst in c_instances: instances.append(f"{inst[:-4]}")
    
    return instances

def get_faith_evaluated_xai_modes(test_id, xai_algorithm, surrogate_model, xai_modes, logger):
    modes_mapping = dict()
    
    if xai_modes is None:
        modes = list()
        for mode in os.listdir(f"{EVAL_ROOT}/faithfulness/{test_id}"):
            if xai_algorithm in mode: modes.append(mode)
    
    else:
        xai_modes = [mode.strip() for mode in xai_modes.split(',')]
        modes = list()
        for mode in xai_modes: modes.append(f"{xai_algorithm}_{mode}_{surrogate_model}")
    
    logger.warning(f"These are the ['{xai_algorithm}', '{surrogate_model}'] modes that were evaluated with Faithfulness")
    logger.warning(f"{modes}")
    
    for i in range(0, len(modes)): modes_mapping[modes[i]] = f"mode{i}"
    return modes_mapping

def collect_all_instances_attr_scores_per_patch_parallel(test_id, instances, modes_mapping, exp_metadata, stability_eval_name):
    args = [(test_id, inst, modes_mapping, exp_metadata, stability_eval_name) for inst in instances]
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(process_instance_attr_scores_per_patch, args)

def process_instance_attr_scores_per_patch(args):
    test_id, inst, modes_mapping, exp_metadata, stability_eval_name = args
    
    if not os.path.exists(f"{EVAL_ROOT}/stability/{test_id}/instances/{inst}/{inst}_all_mode_scores.csv"):
        df = pd.DataFrame()
        for mode in list(modes_mapping.keys()):
            dirname = exp_metadata[f"{mode}_METADATA"]["DIR_NAME"]
            patch_width = exp_metadata[f"{mode}_METADATA"]["PATCH_DIM"]["WIDTH"]
            patch_height = exp_metadata[f"{mode}_METADATA"]["PATCH_DIM"]["HEIGHT"]
            with open(f"{XAI_ROOT}/explanations/patches_{patch_width}x{patch_height}_removal/{dirname}/{inst}/{inst}_scores.json") as f: scores = json.load(f)
        
            patches, attr_scores = list(scores.keys()), list(scores.values())
            if "patches" not in df.columns: df["patches"] = patches
            df[mode] = attr_scores
        
        df.set_index("patches", inplace=True)
        df.to_csv(f"{EVAL_ROOT}/stability/{test_id}/{stability_eval_name}/instances/{inst}/{inst}_all_mode_scores.csv", index=True, header=True)

def get_mode_pairs(modes_mapping):
    modes = list(modes_mapping.keys())
    return list(combinations(modes, 2))

def produce_instances_abs_differences_report_parallel(test_id, instances, modes_mapping, mode_pairs, stability_eval_name):
    args = [(test_id, inst, modes_mapping, mode_pairs, stability_eval_name) for inst in instances]
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(process_instance_abs_differences, args)

def process_instance_abs_differences(args):
    test_id, inst, modes_mapping, mode_pairs, stability_eval_name = args
    
    if not os.path.exists(f"{EVAL_ROOT}/stability/{test_id}/{stability_eval_name}/instances/{inst}/{inst}_abs_differences.csv"):
        all_mode_scores = pd.read_csv(f"{EVAL_ROOT}/stability/{test_id}/{stability_eval_name}/instances/{inst}/{inst}_all_mode_scores.csv", index_col=0, header=0)
        abs_differences = pd.DataFrame()
        abs_differences["patches"] = all_mode_scores.index
        abs_differences.set_index("patches", inplace=True)
        
        for pair in mode_pairs:
            mode1, mode2 = modes_mapping[pair[0]], modes_mapping[pair[1]]
            for patch in abs_differences.index:
                score1, score2 = all_mode_scores.at[patch, pair[0]], all_mode_scores.at[patch, pair[1]]
                abs_differences.at[patch, f"{mode1}X{mode2}"] = np.abs(score1 - score2)
                
        for patch in abs_differences.index:
            mean, std = np.mean(abs_differences.loc[patch]), np.std(abs_differences.loc[patch])
            abs_differences.at[patch, "mean"] = mean
            abs_differences.at[patch, "std"] = std
        
        abs_differences.to_csv(f"{EVAL_ROOT}/stability/{test_id}/{stability_eval_name}/instances/{inst}/{inst}_abs_differences.csv", index=True, header=True)

def compute_global_abs_differences_stats(test_id, instances, stability_eval_name):
    global_statistics = pd.DataFrame(columns=["Instance"])
    global_statistics.set_index("Instance", inplace=True)
    
    for instance in instances:
        abs_differences = pd.read_csv(f"{EVAL_ROOT}/stability/{test_id}/{stability_eval_name}/instances/{instance}/{instance}_abs_differences.csv", index_col=0, header=0)
        global_statistics.at[instance, "Mode Pairs Evaluated"] = len(abs_differences.columns) - 2 # Exclude mean and std columns
        global_statistics.at[instance, "Global_Mean_Abs_Difference"] = np.mean(abs_differences["mean"])
        global_statistics.at[instance, "Global_Std_Abs_Difference"] = np.std(abs_differences["mean"])
        global_statistics.at[instance, "Global_Median_Abs_Difference"] = np.median(abs_differences["mean"])
        global_statistics.at[instance, "Global_Max_Abs_Difference"] = np.max(abs_differences["mean"])
        global_statistics.at[instance, "Global_Min_Abs_Difference"] = np.min(abs_differences["mean"])
    
    global_statistics.to_csv(f"{EVAL_ROOT}/stability/{test_id}/{stability_eval_name}/abs_differences_global_statistics.csv", index=True, header=True)
    
def produce_instances_jaccard_report_parallel(test_id, instances, modes_mapping, mode_pairs, stability_eval_name, logger):
    rules = ["best", "worst"]
    ratios = [0.01, 0.05, 0.1, 0.25]
    
    for rule in rules:
        for ratio in ratios:
            output_path = f"{EVAL_ROOT}/stability/{test_id}/{stability_eval_name}/jaccard_report_{rule}_{ratio}.csv"
            if not os.path.exists(output_path):
                args = [(test_id, inst, modes_mapping, mode_pairs, rule, ratio, stability_eval_name) for inst in instances]
                
                with mp.Pool(mp.cpu_count()) as pool:
                    results = pool.map(process_instance_jaccard, args)
                
                jaccard_report = pd.DataFrame(columns=["Instance"])
                jaccard_report["Instance"] = instances
                jaccard_report.set_index("Instance", inplace=True)
                for inst, values in results:
                    for k, v in values.items():
                        jaccard_report.at[inst, k] = v
            
            jaccard_report.to_csv(output_path, index=True, header=True)
            logger.info(f"Produced Jaccard Similarity Report: {rule}_{ratio}")
            
def process_instance_jaccard(args):
    test_id, inst, modes_mapping, mode_pairs, rule, ratio, stability_eval_name = args
    
    all_mode_scores_path = f"{EVAL_ROOT}/stability/{test_id}/{stability_eval_name}/instances/{inst}/{inst}_all_mode_scores.csv"
    all_mode_scores = pd.read_csv(all_mode_scores_path, index_col=0)
    total_patches = len(all_mode_scores.index)
    num_patches = int(total_patches * ratio)

    row = {}
    for pair in mode_pairs:
        mode1, mode2 = modes_mapping[pair[0]], modes_mapping[pair[1]]
        mode1_scores = all_mode_scores[pair[0]]
        mode2_scores = all_mode_scores[pair[1]]
        
        if rule == "best":
            mode1_sorted = mode1_scores.sort_values(ascending=False)
            mode2_sorted = mode2_scores.sort_values(ascending=False)
        else:
            mode1_sorted = mode1_scores.sort_values(ascending=True)
            mode2_sorted = mode2_scores.sort_values(ascending=True)
        
        mode1_filtered = mode1_sorted.head(num_patches).index.tolist()
        mode2_filtered = mode2_sorted.head(num_patches).index.tolist()
        
        intersection = len(set(mode1_filtered) & set(mode2_filtered))
        union = len(set(mode1_filtered) | set(mode2_filtered))
        jaccard = intersection / union 
        row[f"{mode1}X{mode2}"] = jaccard
 
    return inst, row