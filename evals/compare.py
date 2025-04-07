import torch
from argparse import ArgumentParser

from evals.utils.compare_utils import pair_confidence_test, pair_explanations_test

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-base_id", type=str, required=True)
    parser.add_argument("-ret_id", type=str, required=True)
    parser.add_argument("-subject", type=str, required=True, choices=["confidence", "explanations"])
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase", "GLimeBinomial"])
    parser.add_argument("-xai_mode", type=str, default="base", choices=["base", "counterfactual_top_class"])
    
    return parser.parse_args()

def validate_args(args) -> bool:
    BASELINE, RETRAINED = args.base_id, args.ret_id
    RET_BASE, _ = RETRAINED.split(':')
    
    if RET_BASE != BASELINE: return False
    else: return True

### #### ###
### MAIN ###
### #### ###
if __name__ == '__main__':
    args = get_args()
    
    if not validate_args(args):
        print("*** THERE ARE INCONGRUENCES WITHIN THE TEST TO BE COMPARED! ***")
        exit(1)
    
    BASELINE_ID, RETRAINED_ID = args.base_id, args.ret_id
    SUBJECT = args.subject
    XAI_ALGORITHM = args.xai_algorithm
    XAI_MODE = args.xai_mode
    
    print(f"Baseline Experiment: {BASELINE_ID}")
    print(f"Re-Trained Experiment: {RETRAINED_ID}")
    
    if SUBJECT == "confidence":
        print("*** BEGINNING OF CONFIDENCE PAIR TEST ***")
        pair_confidence_test(BASELINE_ID, RETRAINED_ID)
    if SUBJECT == "explanations":
        print(f"*** BEGINNING OF EXPLANATIONS PAIR TEST -> '{XAI_ALGORITHM}' ALGORITHM") 
        pair_explanations_test(BASELINE_ID, RETRAINED_ID, XAI_ALGORITHM, XAI_MODE)

    print()
    torch.cuda.empty_cache()