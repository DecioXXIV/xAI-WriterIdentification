import torch
from argparse import ArgumentParser

from evals.utils.compare_utils import pair_confidence_test, pair_explanations_test

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-base_id", type=str, required=True)
    parser.add_argument("-ret_id", type=str, required=True)
    parser.add_argument("-subject", type=str, required=True, choices=["confidence", "explanations"])
    parser.add_argument("-mode", type=str, choices=["scan", "random"])
    parser.add_argument("-mult_factor", type=str, default="1")
    parser.add_argument("-iters", type=int, default=3)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase"])
    
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
    
    BASELINE, RETRAINED = args.base_id, args.ret_id
    SUBJECT = args.subject
    MODE = args.mode
    XAI_ALGORITHM = args.xai_algorithm
    
    MULT_FACTORS = list()
    for f in args.mult_factor.split(','): MULT_FACTORS.append(int(f))
    ITERS = args.iters
    
    print(f"Baseline Experiment: {BASELINE}")
    print(f"Re-Trained Experiment: {RETRAINED}")
    
    if SUBJECT == "confidence":
        if MODE == "scan": print(f"*** BEGINNING OF CONFIDENCE PAIR TEST -> '{MODE}' MODE, MULT_FACTOR = {MULT_FACTORS} ***")
        if MODE == "random": print(f"*** BEGINNING OF CONFIDENCE PAIR TEST -> '{MODE}' MODE, ITERS = {ITERS} ***")
        pair_confidence_test(BASELINE, RETRAINED, MODE, MULT_FACTORS, ITERS)
    if SUBJECT == "explanations":
        print(f"*** BEGINNING OF EXPLANATIONS PAIR TEST -> '{XAI_ALGORITHM}' ALGORITHM") 
        pair_explanations_test(BASELINE, RETRAINED, XAI_ALGORITHM)

    print()
    torch.cuda.empty_cache()