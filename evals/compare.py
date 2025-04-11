import torch
from argparse import ArgumentParser

from utils import get_logger

from evals.utils.compare_utils import pair_confidence_test, pair_explanations_test

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-base_id", type=str, required=True)
    parser.add_argument("-ret_id", type=str, required=True)
    parser.add_argument("-subject", type=str, required=True, choices=["confidence", "explanations"])
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase", "GLimeBinomial"])
    parser.add_argument("-xai_mode", type=str, default="base")
    
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
    BASELINE_ID, RETRAINED_ID = args.base_id, args.ret_id
    SUBJECT = args.subject
    XAI_ALGORITHM = args.xai_algorithm
    XAI_MODE = args.xai_mode
    logger = get_logger(BASELINE_ID)
    
    if not validate_args(args):
        logger.error("*** THERE ARE INCONGRUENCES WITHIN THE TEST TO BE COMPARED! ***")
        exit(1)
    
    logger.info(f"Baseline Experiment: {BASELINE_ID}")
    logger.info(f"Re-Trained Experiment: {RETRAINED_ID}")
    
    if SUBJECT == "confidence":
        logger.info("*** BEGINNING OF CONFIDENCE PAIR TEST ***")
        pair_confidence_test(BASELINE_ID, RETRAINED_ID, logger)
    if SUBJECT == "explanations":
        logger.info(f"*** BEGINNING OF EXPLANATIONS PAIR TEST -> '{XAI_ALGORITHM}' ALGORITHM") 
        pair_explanations_test(BASELINE_ID, RETRAINED_ID, XAI_ALGORITHM, XAI_MODE, logger)

    torch.cuda.empty_cache()