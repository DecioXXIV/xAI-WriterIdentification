import os, json
import multiprocessing as mp
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import kendalltau

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
    differences_report = pd.DataFrame(columns=["Instance"])
    differences_report.set_index("Instance", inplace=True)
    tasks = [(test_id, inst, pair, modes_mapping, stability_eval_name) for pair in mode_pairs for inst in instances]
    
    with mp.Pool(mp.cpu_count()) as pool: results = pool.map(process_instance_abs_differences, tasks)

    for inst, col_name, value in results: differences_report.at[inst, col_name] = value

    differences_report.to_csv(f"{EVAL_ROOT}/stability/{test_id}/{stability_eval_name}/euclidean_distances.csv", index=True, header=True)

def process_instance_abs_differences(args):
    test_id, inst, pair, modes_mapping, stability_eval_name = args
    mode1, mode2 = modes_mapping[pair[0]], modes_mapping[pair[1]]
    file_path = f"{EVAL_ROOT}/stability/{test_id}/{stability_eval_name}/instances/{inst}/{inst}_all_mode_scores.csv"
    
    all_mode_scores = pd.read_csv(file_path, index_col=0, header=0)
    mode1_scores = np.array(all_mode_scores[pair[0]].values)
    mode2_scores = np.array(all_mode_scores[pair[1]].values)
    distance = np.linalg.norm(np.abs(mode1_scores - mode2_scores))

    return (inst, f"{mode1}X{mode2}", distance)

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

def produce_instances_kendalltau_report_parallel(test_id, instances, modes_mapping, mode_pairs, stability_eval_name, logger):
    ratios = [0.01, 0.05, 0.1, 0.25]

    for ratio in ratios:
        output_path = f"{EVAL_ROOT}/stability/{test_id}/{stability_eval_name}/kendalltau_report_best_{ratio}.csv"
        if not os.path.exists(output_path):
            args = [(test_id, inst, modes_mapping, mode_pairs, ratio, stability_eval_name) for inst in instances]

            with mp.Pool(mp.cpu_count()) as pool: results = pool.map(process_instance_kendalltau, args)

            kendalltau_report = pd.DataFrame(columns=["Instance"])
            kendalltau_report.set_index("Instance", inplace=True)
            for inst, values in results:
                for k, v in values.items():
                    kendalltau_report.at[inst, k] = v
                
            kendalltau_report.to_csv(output_path, index=True, header=True)
            logger.info(f"Produced Kendall-Tau Similarity Report: best_{ratio}")

def process_instance_kendalltau(args):
    test_id, inst, modes_mapping, mode_pairs, ratio, stability_eval_name = args
    
    all_mode_scores_path = f"{EVAL_ROOT}/stability/{test_id}/{stability_eval_name}/instances/{inst}/{inst}_all_mode_scores.csv"
    all_mode_scores = pd.read_csv(all_mode_scores_path, index_col=0)
    total_patches = len(all_mode_scores.index)
    num_patches = int(total_patches * ratio)

    row = {}
    for pair in mode_pairs:
        mode1, mode2 = modes_mapping[pair[0]], modes_mapping[pair[1]]
        mode1_scores = all_mode_scores[pair[0]]
        mode2_scores = all_mode_scores[pair[1]]
        
        mode1_keys_toreport = mode1_scores.head(num_patches).index.tolist()
        mode2_keys_toreport = mode2_scores.head(num_patches).index.tolist()
        all_keys_toreport = list(set(mode1_keys_toreport) | set(mode2_keys_toreport))

        mode1_filtered_scores, mode2_filtered_scores = dict(), dict()
        for k in all_keys_toreport:
            mode1_filtered_scores[k] = mode1_scores.loc[k]
            mode2_filtered_scores[k] = mode2_scores.loc[k]

        mode1_filtered_ordered_keys = list(dict(sorted(mode1_filtered_scores.items(), key=lambda item: item[1], reverse=True)).keys())
        mode2_filtered_ordered_keys = list(dict(sorted(mode2_filtered_scores.items(), key=lambda item: item[1], reverse=True)).keys())

        kt = kendalltau(mode1_filtered_ordered_keys, mode2_filtered_ordered_keys)[0]
        row[f"{mode1}X{mode2}"] = float(kt)
 
    return inst, row  
