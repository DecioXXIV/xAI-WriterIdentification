import torch, os, pickle
from image_utils import load_rgb_mean_std
from model_utils import *
from explanations_utils import prepare_instances_to_explain
from PIL import Image
from datetime import datetime
from lime_base_explainer import LimeBaseExplainer, get_crops_bbxs
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
    parser.add_argument("-remove_patches", type=str, default="False")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = get_args()
    TEST_ID = args.test_id
    MODEL_TYPE = args.model
    BLOCK_WIDTH, BLOCK_HEIGHT, CROP_SIZE = args.block_width, args.block_height, args.crop_size
    SURROGATE_MODEL = args.surrogate_model
    LIME_ITERS = args.lime_iters
    REMOVE_PATCHES = args.remove_patches

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
    
    classes = os.listdir(root + f"/tests/{TEST_ID}/test")
    num_classes = len(classes)
    
    print("Classes:", classes)
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

    mp_explainer = LimeBaseExplainer(classifier=MODEL_NAME, 
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
    instances, labels = list(), list()
    
    class_to_idx = None
    with open(root + f"/tests/{TEST_ID}/output/class_to_idx.pkl", "rb") as f:
        class_to_idx = pickle.load(f)
    
    if "ret" not in TEST_ID:
        train_instances, train_labels = prepare_instances_to_explain(root, TEST_ID, classes, class_to_idx, "train", CWD)
        instances.extend(train_instances)
        labels.extend(train_labels)
    
    test_instances, test_labels = prepare_instances_to_explain(root, TEST_ID, classes, class_to_idx, "test", CWD)
    instances.extend(test_instances)
    labels.extend(test_labels)
        
    # 2 -> Explanation Process
    overlap = CROP_SIZE - 25

    for instance_name, label in zip(instances, labels):
        print(f"Processing Instance '{instance_name}' with label '{label}'")
        
        img_path, mask_path = os.path.join(CWD, "data", instance_name), os.path.join(CWD, "def_mask.png")
        img, mask = Image.open(img_path), Image.open(mask_path)

        # 3.1 -> Feature Attribution for the Instance Superpixels (identified by the Page Mask)
        mp_explainer.compute_superpixel_scores(img, mask, instance_name, label, LIME_ITERS, CROP_SIZE, overlap)
        mp_explainer.visualize_superpixel_scores_outcomes(img, mask, instance_name, reduction_method="mean", min_eval=2)

        # 3.2 -> Generating the Explanation for the Instance Crops by removing "Relevant", "Misleading" and "Random" Patches
        if REMOVE_PATCHES == "True":
            crops_bbxs = get_crops_bbxs(img, final_width=CROP_SIZE, final_height=CROP_SIZE)
            mp_explainer.compute_masked_patches_explanation(img, mask, instance_name, label, crops_bbxs, CROP_SIZE, reduction_method="mean", min_eval=10, num_samples_for_baseline=10)
        
        os.system(f"rm ./data/{instance_name}")
        
    os.system(f"cp {root}/tests/{TEST_ID}/train/rgb_stats.pkl ./explanations/patches_{BLOCK_WIDTH}x{BLOCK_HEIGHT}_removal/{dir_name}/rgb_stats.pkl")