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

def get_faith_evaluated_xai_modes(test_id, xai_algorithm, surrogate_model, logger):
    modes_mapping = dict()
    modes = list()
    for mode in os.listdir(f"{EVAL_ROOT}/faithfulness/{test_id}"):
        if xai_algorithm in mode: modes.append(mode)
    logger.warning(f"These are the ['{xai_algorithm}', '{surrogate_model}'] modes that were evaluated with Faithfulness")
    logger.warning(f"{modes}")
    
    for i in range(0, len(modes)): modes_mapping[modes[i]] = f"mode{i}"
    return modes_mapping

def collect_all_instances_attr_scores_per_patch_parallel(test_id, instances, modes_mapping, exp_metadata):
    args = [(test_id, inst, modes_mapping, exp_metadata) for inst in instances]
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(process_instance_attr_scores_per_patch, args)

def process_instance_attr_scores_per_patch(args):
    test_id, inst, modes_mapping, exp_metadata = args
    
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
        df.to_csv(f"{EVAL_ROOT}/stability/{test_id}/instances/{inst}/{inst}_all_mode_scores.csv", index=True, header=True)

def get_mode_pairs(modes_mapping):
    modes = list(modes_mapping.keys())
    return list(combinations(modes, 2))

def produce_instances_abs_differences_report_parallel(test_id, instances, modes_mapping, mode_pairs):
    pass
    args = [(test_id, inst, modes_mapping, mode_pairs) for inst in instances]
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(process_instance_abs_differences, args)

def process_instance_abs_differences(args):
    test_id, inst, modes_mapping, mode_pairs = args
    
    if not os.path.exists(f"{EVAL_ROOT}/stability/{test_id}/instances/{inst}/{inst}_abs_differences.csv"):
        all_mode_scores = pd.read_csv(f"{EVAL_ROOT}/stability/{test_id}/instances/{inst}/{inst}_all_mode_scores.csv", index_col=0, header=0)
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
        
        abs_differences.to_csv(f"{EVAL_ROOT}/stability/{test_id}/instances/{inst}/{inst}_abs_differences.csv", index=True, header=True)
        
        final_stats = dict()
        final_stats["whole_mean"] = np.mean(abs_differences["mean"])
        final_stats["whole_std"] = np.std(abs_differences["std"])
        
        with open(f"{EVAL_ROOT}/stability/{test_id}/instances/{inst}/{inst}_whole_abs_differences_stats.json", "w") as f:
            json.dump(final_stats, f, indent=4)
        