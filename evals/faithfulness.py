import os, json, torch
import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score

from xai.image_masker import ImageMasker
from classifiers.classifier_NN.model import NN_Classifier
from classifiers.classifier_NN.utils import Train_Test_DataLoader, load_rgb_mean_std, process_test_set

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
MODEL_ROOT = "./classifiers"
XAI_ROOT = "./xai"
EVAL_ROOT = "./evals"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-xai_algorithm", type=str, required=True, choices=["LimeBase"])
    parser.add_argument("-mask_ceil", type=float, required=True)
    parser.add_argument("-mask_step", type=float, required=True)
    parser.add_argument("-test_iterations", type=int, default=1)
    
    return parser.parse_args()

def load_metadata(metadata_path) -> dict:
    try:
        with open(metadata_path, 'r') as jf: return json.load(jf)
    except json.JSONDecodeError as e: raise ValueError(f"Error occurred while decoding JSON file '{metadata_path}': {e}")
    except Exception as e: raise FileNotFoundError(f"Error occurred while reading metadata file '{metadata_path}': {e}")

def get_test_instances_to_mask(dataset, classes):
    dataset_dir = f"{DATASET_ROOT}/{dataset}"
    instances, instance_full_paths = list(), list()
    
    for c in classes:
        class_instances = list()
        if dataset == "CEDAR_Letter":
            class_instances = [inst for inst in os.listdir(f"{dataset_dir}/{c}") if "c" in inst]
        if dataset == "CVL":
            class_instances = [inst for inst in os.listdir(f"{dataset_dir}/{c}") if ("-3" in inst or "-7" in inst)]
        if dataset == "VatLat653":
            class_instances = [inst for inst in os.listdir(f"{dataset_dir}/{c}") if "t" in inst]
        
        for i in class_instances:
            instances.append(i)
            instance_full_paths.append(f"{dataset_dir}/{c}/{i}")
    
    return instances, instance_full_paths

def mask_test_instances(instances, paths, test_id, exp_dir, mask_rate, block_width, block_height, xai_algorithm, exp_metadata):
    masker = ImageMasker(
        inst_set="test",
        instances=instances,
        paths=paths,
        test_id=test_id,
        exp_dir=exp_dir,
        mask_rate=mask_rate,
        mode="saliency",
        block_width=block_width,
        block_height=block_height,
        xai_algorithm=xai_algorithm,
        exp_metadata=exp_metadata
    )
    
    masker()
    
    m_rate2instance = EXP_METADATA[f"MASK_PROCESS_test_saliency_{mask_rate}_{xai_algorithm}_METADATA"]["FULL_INSTANCES"]
    total_instances, bad_instances = len(m_rate2instance), 0
    for _, masked_area in m_rate2instance.items():
        if masked_area < mask_rate: bad_instances += 1
    
    if bad_instances < 0.33 * total_instances: return True
    else: return False

def test_model(model_type, test_id, classes, xai_algorithm, m_rate):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_CP = f"{MODEL_ROOT}/cp/Test_3_TL_val_best_model.pth"
    TEST_DIR = f"{MODEL_ROOT}/classifier_{model_type}/tests/{test_id}"
    CP_TO_TEST = f"{MODEL_ROOT}/classifier_{model_type}/tests/{test_id}/output/checkpoints/Test_{test_id}_MLC_val_best_model.pth"
    
    if model_type == "NN": model = NN_Classifier(num_classes=len(classes), mode='frozen', cp_path=BASE_CP)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(CP_TO_TEST)['model_state_dict'])
    model.eval()
    
    directory = f"{EVAL_ROOT}/{test_id}/test_set_masked_saliency_{m_rate}_{xai_algorithm}"
    mean_, std_ = load_rgb_mean_std(f"{TEST_DIR}")
    
    dl = Train_Test_DataLoader(directory=directory, classes=classes, batch_size=1, img_crop_size=380, weighted_sampling=False, phase='test', mean=mean_, std=std_, shuffle=True)
    
    _, labels, preds, _, idx_to_c = process_test_set(dl, DEVICE, model)
    label_class_names = [idx_to_c[id_] for id_ in labels]
    pred_class_names = [idx_to_c[id_] for id_ in preds]
    
    return accuracy_score(label_class_names, pred_class_names)
    

### #### ###
### MAIN ###
### #### ###
if __name__ == '__main__':
    args = get_args()
    TEST_ID = args.test_id
    EXP_METADATA_PATH = f"{LOG_ROOT}/{TEST_ID}-metadata.json"
    
    try: EXP_METADATA = load_metadata(EXP_METADATA_PATH)
    except Exception as e:
        print(e)
        exit(1)
    
    XAI_ALGORITHM = args.xai_algorithm
    MASK_CEIL, MASK_STEP = args.mask_ceil, args.mask_step
    TEST_ITERATIONS = args.test_iterations
    
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]
    DATASET = EXP_METADATA["DATASET"]
    CLASSES = list(EXP_METADATA["CLASSES"].keys())
    EXP_DIR = EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["DIR_NAME"]
    BLOCK_WIDTH = EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["BLOCK_DIM"]["WIDTH"]
    BLOCK_HEIGHT = EXP_METADATA[f"{XAI_ALGORITHM}_METADATA"]["BLOCK_DIM"]["HEIGHT"]
    
    instances, paths = get_test_instances_to_mask(DATASET, CLASSES)
    
    mask_rates = np.arange(start=0.0, stop=MASK_CEIL+MASK_STEP, step=MASK_STEP)
    performances = list()
    
    os.makedirs(f"{EVAL_ROOT}/{TEST_ID}", exist_ok=True)
    with open(f"{EVAL_ROOT}/{TEST_ID}/faithfulness.txt", 'w') as f: f.write(f"*** FAITHFULNESS COMPUTATION FOR TEST: {TEST_ID} ***\n")
    
    for i, m_rate in enumerate(mask_rates):
        print(f"BEGINNING OF ACCURACY COMPUTATION FOR M_RATE: {m_rate}")
        if m_rate == 0.0:
            os.makedirs(f"{EVAL_ROOT}/{TEST_ID}/test_set_masked_saliency_{m_rate}_{XAI_ALGORITHM}", exist_ok=True)
            for inst, src in zip(instances, paths):
                c = src.split('/')[-2]
                os.makedirs(f"{EVAL_ROOT}/{TEST_ID}/test_set_masked_saliency_{m_rate}_{XAI_ALGORITHM}/{c}", exist_ok=True)
                dest = f"{EVAL_ROOT}/{TEST_ID}/test_set_masked_saliency_{m_rate}_{XAI_ALGORITHM}/{c}/{inst}"
                os.system(f"cp {src} {dest}")
            mask_condition = True
        
        else: mask_condition = mask_test_instances(instances, paths, TEST_ID, EXP_DIR, m_rate, BLOCK_WIDTH, BLOCK_HEIGHT, XAI_ALGORITHM, EXP_METADATA)
        
        if mask_condition:
            m_rate_performances = np.zeros(TEST_ITERATIONS)
            for iter in range(0, TEST_ITERATIONS):
                m_rate_performances[iter] = test_model(MODEL_TYPE, TEST_ID, CLASSES, XAI_ALGORITHM, m_rate)
            
            mean_perf = np.mean(m_rate_performances)
            var_perf = np.var(m_rate_performances)
            print(f"MEAN ACCURACY ON {TEST_ITERATIONS} ITERATIONS FOR M_RATE {m_rate}: {mean_perf}\n")
            
            with open(f"{EVAL_ROOT}/{TEST_ID}/faithfulness.txt", 'a') as f:
                f.write(f"M_RATE: {m_rate}; ITERATIONS: {TEST_ITERATIONS}\n")
                f.write(f"Mean Accuracy: {mean_perf}; Variance: {var_perf}\n")
                f.write("### ------------------ ###\n\n")
            
            performances.append(mean_perf)
        else:
            print(f"Faithfulness Computation stopped on m_rate: {m_rate}")
            break
    
    performances = np.array(performances)
    performances = (performances - performances[0]) * -MASK_STEP
    
    faithfulness = np.sum(performances) / (len(mask_rates) * MASK_STEP)
    
    print(f"Faithfulness for {TEST_ID}: {faithfulness}")
    
    with open(f"{EVAL_ROOT}/{TEST_ID}/faithfulness.txt", 'a') as f:
        f.write("### ------------------ ###\n")
        f.write(f"Faithfulness: {faithfulness}")