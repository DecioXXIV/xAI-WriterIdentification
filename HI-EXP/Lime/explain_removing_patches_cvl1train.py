import torch, os
from image_utils import load_rgb_mean_std
from model_utils import *
from explanations_utils import generate_page_mask
from PIL import Image
from datetime import datetime
from masked_patches_explainer import MaskedPatchesExplainer, get_crops_bbxs
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Hyperparameters for the Explanation process", add_help=True)
    parser.add_argument("-test_id", type=str)
    parser.add_argument("-model", type=str)
    parser.add_argument("-block_width", type=int)
    parser.add_argument("-block_height", type=int)
    parser.add_argument("-crop_size", type=int)
    parser.add_argument("-surrogate_model", type=str)
    parser.add_argument("-lime_iters", type=int)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = get_args()
    TEST_ID = args.test_id
    MODEL_TYPE = args.model
    OPT = args.opt
    BLOCK_WIDTH, BLOCK_HEIGHT, CROP_SIZE = args.block_width, args.block_height, args.crop_size
    SURROGATE_MODEL = args.surrogate_model
    LIME_ITERS = args.lime_iters

    assert MODEL_TYPE in ['NN', 'SVM', 'GB'], "Model must be in ['NN', 'SVM', 'GB']"
    assert SURROGATE_MODEL in ['LinReg', 'Ridge'], "Surrogate Model must be in ['LinReg', 'Ridge']"

    MODEL_NAME = None
    match MODEL_TYPE:
        case 'NN':
            MODEL_NAME = "classifier_NN"
        case 'SVM':
            MODEL_NAME = "classifier_SVM"
        case 'GB':
            MODEL_NAME = "classifier_GB"

    CWD = os.getcwd()
    root = CWD + f"/./../{MODEL_NAME}"
    cp_base = root + "/cp/Test_3_TL_val_best_model.pth"
    cp = root + f"/tests/{TEST_ID}/output/checkpoints/Test_{TEST_ID}_MLC_val_best_model.pth"

    # Model Loading
    print("LOADING MODEL...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        print("Device:", torch.cuda.get_device_name(0))
    else:
        print("Device: CPU")
    
    num_classes = len(os.listdir(root + f"/tests/{TEST_ID}/test"))
    model = None

    match MODEL_TYPE:
        case 'NN':
            model = NN_Classifier(num_classes=num_classes, mode='frozen', cp_path=cp_base)
            model.load_state_dict(torch.load(cp)['model_state_dict'])
            model.eval()
            model.to(DEVICE)
        case 'SVM':
            model = SVM_Classifier(mode='frozen', cp_path=cp_base)
            model.base_model.to(DEVICE)
            model.base_model.eval()
            model.tail = load_tail(root + f"/tests/{TEST_ID}/output/checkpoints/svm_tail.pkl")
        case 'GB':
            model = GB_Classifier(mode='frozen', cp_path=cp_base)
            model.base_model.to(DEVICE)
            model.base_model.eval()
            model.tail = load_tail(root + f"/tests/{TEST_ID}/output/checkpoints/gb_tail.pkl")

    print("...Model Loaded!\n")

    # Creating MaskedPatchesExplainer
    mean, std = load_rgb_mean_std(root + f"/tests/{TEST_ID}/train")

    now = str(datetime.now())
    now = str.replace(now, '-', '.')
    now = str.replace(now, ' ', '_')
    now = str.replace(now, ':', '.')
    dir_name = TEST_ID + "-" + MODEL_TYPE + "-" + now + "-" + SURROGATE_MODEL

    # dir_name = TEST_ID + "-" + MODEL_TYPE + "-" + "2024.08.22_21.14.56.132395" + "-" + SURROGATE_MODEL

    mp_explainer = MaskedPatchesExplainer(classifier=MODEL_NAME, 
                                          test_id=dir_name, 
                                          block_size=(BLOCK_WIDTH, BLOCK_HEIGHT), 
                                          model=model,
                                          surrogate_model=SURROGATE_MODEL, 
                                          mean=mean, 
                                          std=std, 
                                          device=DEVICE)
    
    ### ################### ###
    ### EXPLANATION PROCESS ###
    ### ################### ###
    files = ["0002-1", "0002-6", "0002-8", "0017-1", "0017-6", "0017-8", "0023-1", "0023-6", "0023-8"]
    instances = list()
    for f in files:
        for i in range(0, 4):
            for j in range(0, 7):
                instances.append(f + "_" + str(i) + "_" + str(j))
    labels = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
              2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]

    # 2 -> Generating the Page Masks (if they don't already exist)
    for instance in instances:
        if not os.path.exists(f"./explanations/page_level/{instance}/{instance}_mask_blocks_{BLOCK_WIDTH}x{BLOCK_HEIGHT}.png"):
            generate_page_mask(instance, BLOCK_WIDTH, BLOCK_HEIGHT)
    
    # 3 -> Explanation Process
    overlap = CROP_SIZE - 25

    for instance, label in zip(instances, labels):
        print(f"Processing Instance '{instance}' with label '{label}'")

        # 3.1 -> Feature Attribution for the Instance Superpixels (identified by the Page Mask)
        mp_explainer.compute_superpixel_scores(instance, label, LIME_ITERS, CROP_SIZE, overlap)
        mp_explainer.visualize_superpixel_scores_outcomes(instance, reduction_method="mean", min_eval=2)

        # 3.2 -> Generating the Explanation for the Instance Crops by removing "Relevant", "Misleading" and "Random" Patches
        img = Image.open(f"./data/{instance}.png")
        crops_bbxs = get_crops_bbxs(img, final_width=CROP_SIZE, final_height=CROP_SIZE)
        mp_explainer.compute_masked_patches_explanation(instance, label, crops_bbxs, CROP_SIZE, reduction_method="mean", min_eval=10, num_samples_for_baseline=10)