import os, torch, pickle, json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from argparse import ArgumentParser
from .model import NN_Classifier
from .utils import Train_Test_DataLoader, Trainer, produce_classification_reports, load_rgb_mean_std

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
CLASSIFIER_ROOT = "./classifiers/classifier_NN"

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###
def get_args():
    parser = ArgumentParser(description="Setup of Fine-tuning hyperparameters")
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-crop_size", type=int, required=True)
    parser.add_argument("-opt", type=str, required=True)
    parser.add_argument("-lr", type=float, required=True)
    parser.add_argument("-train_replicas", type=int, required=True)
    parser.add_argument("-random_seed", type=int, required=True)
    parser.add_argument("-epochs", type=int, default=50)
    return parser.parse_args()

def load_metadata(metadata_path) -> dict:
    try:
        with open(metadata_path, 'r') as jf: return json.load(jf)
    except json.JSONDecodeError as e: raise ValueError(f"Error occurred while decoding JSON file '{metadata_path}': {e}")
    except Exception as e: raise FileNotFoundError(f"Error occurred while reading metadata file '{metadata_path}': {e}")

def create_directories(root_folder, classes):
    os.makedirs(root_folder, exist_ok=True)
    
    for phase in ["train", "val", "test"]:
        for c in classes:
            os.makedirs(f"{root_folder}/{phase}/{c}", exist_ok=True)
    
    os.makedirs(f"{root_folder}/output", exist_ok=True)

def split_and_copy_files(dataset, source_dir, class_name, train_replicas, random_seed):
    files = os.listdir(source_dir)
    train, test = None, None
    
    if dataset == "CEDAR_Letter":
        train = [f for f in files if "c" not in f]
        test = [f for f in files if "c" in f]
    if dataset == "CVL":
        train = [f for f in files if ("-3" not in f and "-7" not in f)]
        test = [f for f in files if ("-3" in f or "-7" in f)]
        
    np.random.seed(random_seed)
    np.random.shuffle(test)
    val = np.random.choice(test, size=int(len(test)/3), replace=False)
    
    for file in train:
        for i in range(0, train_replicas):
            os.system(f"cp {source_dir}/{file} {EXP_DIR}/train/{class_name}/{file[:-4]}_cp{i+1}{file[-4:]}")
        
    for file in val:
        os.system(f"cp {source_dir}/{file} {EXP_DIR}/val/{class_name}/{file[:-4]}{file[-4:]}")

    for file in test:
        os.system(f"cp {source_dir}/{file} {EXP_DIR}/test/{class_name}/{file[:-4]}{file[-4:]}")
        
    class_instances = len(os.listdir(f"{EXP_DIR}/train/{class_name}"))
    print(f"Class {c} -> Train Instances: {class_instances}")

def plot_metric(metric):
    values = {"train": [], "val": []}
    
    for phase in values.keys():
        with open(f'{OUTPUT_DIR}/Test_{TEST_ID}_MLC_{phase}_{metric}.pkl', 'rb') as f:
            values[phase] = pickle.load(f)
    
    best_train_metric, best_val_metric = None, None
    if metric == "loss": best_train_metric, best_val_metric = np.min(values['train']), np.min(values['val'])
    if metric == "accuracy": best_train_metric, best_val_metric = np.max(values['train']), np.max(values['val'])
    best_train_epoch = np.where(np.array(values['train']) == best_train_metric)[0][0] + 1
    best_val_epoch = np.where(np.array(values['val']) == best_val_metric)[0][0] + 1
    
    with open(f'{OUTPUT_DIR}/Test_{TEST_ID}_MLC_{metric}.txt', 'w') as f:
        f.write(f"The optimal value of {metric} for the training set is: {round(best_train_metric, 3)}\n")
        f.write(f"The optimal value of {metric} for the validation set is: {round(best_val_metric, 3)}\n")
        f.write(f"Epoch corresponding to the optimal value of the training {metric}: {best_train_epoch}\\{len(values['train'])}\n")
        f.write(f"Epoch corresponding to the optimal value of the validation {metric}: {best_val_epoch}\\{len(values['val'])}\n")
    
    plt.plot(values['train'])
    plt.plot(values['val'])
    plt.title(f"Model {metric}")
    plt.ylabel(f"{metric} [-]")
    plt.xlabel("Epoch [-]")
    plt.legend(['Training', 'Validation'], loc='best')
    plt.savefig(f'{OUTPUT_DIR}/Test_{TEST_ID}_MLC_{metric}.png')
    plt.close()

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
    
    if "FINE_TUNING_TIMESTAMP" in EXP_METADATA:
        print("*** IN RELATION TO THE SPECIFIED EXPERIMENT, THE MODEL HAS ALREADY BEEN FINE-TUNED! ***\n")
        exit(1)
        
    print("*** BEGINNING OF FINE-TUNING PROCESS ***")
    
    BASE_ID, RET_ID = None, None
    try: BASE_ID, RET_ID = TEST_ID.split(':')
    except: BASE_ID = TEST_ID
    
    DATASET = EXP_METADATA["DATASET"]
    CLASSES_DATA = EXP_METADATA["CLASSES"]
    
    CROP_SIZE = args.crop_size
    OPT = args.opt
    LR = args.lr
    TRAIN_REPLICAS = args.train_replicas
    RANDOM_SEED = args.random_seed
    EPOCHS = args.epochs
    
    EXP_METADATA["FINE_TUNING_HP"] = {
        "optimizer": OPT,
        "learning_rate": LR,
        "train_replicas": TRAIN_REPLICAS,
        "random_seed": RANDOM_SEED,
        "total_epochs": EPOCHS
    }
    
    with open(EXP_METADATA_PATH, 'w') as jf: json.dump(EXP_METADATA, jf, indent=4)
    
    os.makedirs(f"{CLASSIFIER_ROOT}/tests", exist_ok=True)

    SOURCE_DATA_DIR = f"{DATASET_ROOT}/{DATASET}/processed"
    EXP_DIR = f"{CLASSIFIER_ROOT}/tests/{TEST_ID}"
    OUTPUT_DIR = f"{CLASSIFIER_ROOT}/tests/{TEST_ID}/output"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = f"{CLASSIFIER_ROOT}/../cp/Test_3_TL_val_best_model.pth"
    
    create_directories(EXP_DIR, CLASSES_DATA.keys())
    
    if "FT_DATASET_PREP_TIMESTAMP" not in EXP_METADATA:
        print("PHASE 1 -> DATASET CREATION...")
        for c, c_type in CLASSES_DATA.items():
            class_source, class_dest = None, None
            if c_type == "base": class_source = f"{SOURCE_DATA_DIR}/{c}"
            else: class_source = f"{SOURCE_DATA_DIR}/{c}-{BASE_ID}_NN_{c_type}"
        
            split_and_copy_files(DATASET, class_source, c, TRAIN_REPLICAS, RANDOM_SEED)
    
        EXP_METADATA["FT_DATASET_PREP_TIMESTAMP"] = str(datetime.now())
        with open(EXP_METADATA_PATH, 'w') as jf: json.dump(EXP_METADATA, jf, indent=4)
        print("...Dataset Created!\n")
    
    else: print("Skipping PHASE 1 (Dataset Creation): it's already available!\n")
    
    mean_, std_ = load_rgb_mean_std(f"{EXP_DIR}/train")
    os.system(f"mv {EXP_DIR}/train/rgb_train_stats.pkl {EXP_DIR}/rgb_train_stats.pkl")
    
    print("PHASE 2 -> MODEL FINE-TUNING...")    
    os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)
    torch.backends.cudnn.benchmark = True
    last_cp = None
                
    train_ds = Train_Test_DataLoader(directory=f"{EXP_DIR}/train", classes=list(CLASSES_DATA.keys()), batch_size=8, img_crop_size=CROP_SIZE, weighted_sampling=True, phase='train', mean=mean_, std=std_, shuffle=True)
    val_ds = Train_Test_DataLoader(directory=f"{EXP_DIR}/val", classes=list(CLASSES_DATA.keys()), batch_size=8, img_crop_size=CROP_SIZE, weighted_sampling=False, phase='val', mean=mean_, std=std_, shuffle=False)
    tds, t_dl = train_ds.load_data()
    vds, v_dl = val_ds.load_data()

    model = NN_Classifier(num_classes=len(CLASSES_DATA), mode='frozen', cp_path=MODEL_PATH)
        
    if "EPOCHS_COMPLETED" in EXP_METADATA:
        epochs_completed = EXP_METADATA["EPOCHS_COMPLETED"]
        EPOCHS = EPOCHS - epochs_completed
        print(f"{epochs_completed} epochs have already been completed: the Fine-Tuning process will be ended with the remaining {EPOCHS} epochs")
            
        last_cp_path = f"{OUTPUT_DIR}/checkpoints/Test_{TEST_ID}_MLC_last_checkpoint.pth"
        last_cp = torch.load(last_cp_path)
        model.load_state_dict(last_cp['model_state_dict'])
         
    model = model.to(DEVICE)
        
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {pytorch_total_params}')
        
    trainer = Trainer(model=model, t_set=t_dl, v_set=v_dl, DEVICE=DEVICE, model_path=OUTPUT_DIR, 
                          history_path=OUTPUT_DIR, exp_metadata=EXP_METADATA, last_cp=last_cp)
    trainer()
        
    EXP_METADATA["FINE_TUNING_TIMESTAMP"] = str(datetime.now())
    with open(EXP_METADATA_PATH, 'w') as jf: json.dump(EXP_METADATA, jf, indent=4)
    print("...Model Fine-Tuning completed!\n")
        
    print("PHASE 3 -> MODEL TESTING...")
    for metric in ["loss", "accuracy"]: plot_metric(metric)
        
    cp_to_test = f"{OUTPUT_DIR}/checkpoints/Test_{TEST_ID}_MLC_val_best_model.pth"
    model = NN_Classifier(num_classes=len(CLASSES_DATA), mode='frozen', cp_path=MODEL_PATH)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(cp_to_test)['model_state_dict']) 
    model.eval()

    dl = Train_Test_DataLoader(directory=f"{EXP_DIR}/test", classes=list(CLASSES_DATA.keys()), batch_size=1, img_crop_size=CROP_SIZE, weighted_sampling=False, phase='test', mean=mean_, std=std_, shuffle=True)
    produce_classification_reports(dl, DEVICE, model, OUTPUT_DIR, TEST_ID)
    print("...Testing reports are now available!\n")
    
    print("PHASE 4 -> DATA & METADATA HANDLING...")
    os.system(f"rm -r {EXP_DIR}/train")
    os.system(f"rm -r {EXP_DIR}/val")
    os.system(f"rm -r {EXP_DIR}/test")
    os.system(f"rm -r {OUTPUT_DIR}/checkpoints/Test_{TEST_ID}_MLC_last_checkpoint.pth")

    torch.cuda.empty_cache()
    print("*** END OF FINE-TUNING PROCESS ***\n")