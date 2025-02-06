import os, torch
from datetime import datetime
from argparse import ArgumentParser

from utils import load_metadata, save_metadata, get_train_instance_patterns, get_test_instance_patterns

from .model import NN_Classifier
from .utils.dataloader_utils import Train_Test_DataLoader, load_rgb_mean_std
from .utils.training_utils import Trainer, plot_metric
from .utils.testing_utils import produce_classification_reports

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
    parser.add_argument("-train_dl_mf", type=int, default=1)
    parser.add_argument("-random_seed", type=int, default=24)
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-ft_mode", type=str, default="frozen", choices=["frozen", "full"])
    
    return parser.parse_args()

def create_directories(root_folder, classes):
    os.makedirs(root_folder, exist_ok=True)
    
    for phase in ["train", "test"]:
        for c in classes:
            os.makedirs(f"{root_folder}/{phase}/{c}", exist_ok=True)
    
    os.makedirs(f"{root_folder}/output", exist_ok=True)

def split_and_copy_files(dataset, source_dir, class_name, train_replicas, random_seed):
    files = os.listdir(source_dir)
    train, test = None, None
    
    train_instance_patterns = get_train_instance_patterns()
    test_instance_patterns = get_test_instance_patterns()
    
    train = [f for f in files if train_instance_patterns[dataset](f)]
    test = [f for f in files if test_instance_patterns[dataset](f)]
            
    for file in train:
        for i in range(0, train_replicas):
            os.system(f"cp {source_dir}/{file} {EXP_DIR}/train/{class_name}/{file[:-4]}_cp{i+1}{file[-4:]}")

    for file in test:
        os.system(f"cp {source_dir}/{file} {EXP_DIR}/test/{class_name}/{file[:-4]}{file[-4:]}")
        
    class_n_train_instances = len(os.listdir(f"{EXP_DIR}/train/{class_name}"))
    print(f"Class {c} -> Train Instances: {class_n_train_instances}")

### #### ###
### MAIN ###
### #### ###    
if __name__ == '__main__':
    args = get_args()
    TEST_ID = args.test_id
    EXP_METADATA_PATH = os.path.join(LOG_ROOT, f"{TEST_ID}-metadata.json")

    try: EXP_METADATA = load_metadata(EXP_METADATA_PATH)
    except Exception as e:
        print(e)
        exit(1)
    
    DATASET = EXP_METADATA["DATASET"]
    CLASSES_DATA = EXP_METADATA["CLASSES"]
    
    CROP_SIZE = args.crop_size
    OPT = args.opt
    LR = args.lr
    TRAIN_REPLICAS = args.train_replicas
    TRAIN_DL_MF = args.train_dl_mf
    RANDOM_SEED = args.random_seed
    EPOCHS = args.epochs
    FT_MODE = args.ft_mode
    
    SOURCE_DATA_DIR = f"{DATASET_ROOT}/{DATASET}/processed"
    EXP_DIR = f"{CLASSIFIER_ROOT}/tests/{TEST_ID}"
    OUTPUT_DIR = f"{CLASSIFIER_ROOT}/tests/{TEST_ID}/output"
    MODEL_PATH = f"{CLASSIFIER_ROOT}/../cp/Test_3_TL_val_best_model.pth"
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if "FINE_TUNING_TIMESTAMP" in EXP_METADATA:
        print(f"*** IN RELATION TO THE EXPERIMENT '{TEST_ID}', THE MODEL HAS ALREADY BEEN FINE-TUNED! ***\n")
    
    else:
        print("*** BEGINNING OF FINE-TUNING PROCESS ***")
    
        BASE_ID, RET_ID = None, None
        try: BASE_ID, RET_ID = TEST_ID.split(':')
        except: BASE_ID = TEST_ID
    
        EXP_METADATA["FINE_TUNING_HP"] = {
            "optimizer": OPT,
            "learning_rate": LR,
            "train_replicas": TRAIN_REPLICAS,
            "train_dl_mf": TRAIN_DL_MF,
            "random_seed": RANDOM_SEED,
            "total_epochs": EPOCHS
        }
        save_metadata(EXP_METADATA, EXP_METADATA_PATH)
    
        os.makedirs(f"{CLASSIFIER_ROOT}/tests", exist_ok=True)
        create_directories(EXP_DIR, CLASSES_DATA.keys())
    
        if "FT_DATASET_PREP_TIMESTAMP" not in EXP_METADATA:
            print("PHASE 1 -> DATASET CREATION...")
            for c, c_type in CLASSES_DATA.items():
                class_source, class_dest = None, None
                if c_type == "base": class_source = f"{SOURCE_DATA_DIR}/{c}"
                else: class_source = f"{SOURCE_DATA_DIR}/{c}-{BASE_ID}_NN_{c_type}"
        
                split_and_copy_files(DATASET, class_source, c, TRAIN_REPLICAS, RANDOM_SEED)
    
            EXP_METADATA["FT_DATASET_PREP_TIMESTAMP"] = str(datetime.now())
            save_metadata(EXP_METADATA, EXP_METADATA_PATH)
            print("...Dataset Created!\n")
    
        else: print("Skipping PHASE 1 (Dataset Creation): it's already available!\n")
    
        mean_, std_ = load_rgb_mean_std(f"{EXP_DIR}/train")
        os.rename(f"{EXP_DIR}/train/rgb_train_stats.pkl", f"{EXP_DIR}/rgb_train_stats.pkl")
    
        print("PHASE 2 -> MODEL FINE-TUNING...")    
        os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)
        torch.backends.cudnn.benchmark = True
        last_cp = None
                
        train_ds = Train_Test_DataLoader(directory=f"{EXP_DIR}/train", classes=list(CLASSES_DATA.keys()), batch_size=8, img_crop_size=CROP_SIZE, mult_factor=TRAIN_DL_MF, weighted_sampling=True, phase='train', mean=mean_, std=std_, shuffle=True)
        val_ds = Train_Test_DataLoader(directory=f"{EXP_DIR}/test", classes=list(CLASSES_DATA.keys()), batch_size=8, img_crop_size=CROP_SIZE, weighted_sampling=False, phase='val', mean=mean_, std=std_, shuffle=False)
        tds, t_dl = train_ds.load_data()
        vds, v_dl = val_ds.load_data()

        model = NN_Classifier(num_classes=len(CLASSES_DATA), mode=FT_MODE, cp_path=MODEL_PATH)
    
        if "refine" in TEST_ID:
            BASE_ID, _ = TEST_ID.split(':')
            last_cp_path = f"{CLASSIFIER_ROOT}/tests/{BASE_ID}/output/checkpoints/Test_{BASE_ID}_MLC_val_best_model.pth"
            last_cp = torch.load(last_cp_path)
            model.load_state_dict(last_cp['model_state_dict'])
        
        if "EPOCHS_COMPLETED" in EXP_METADATA:
            epochs_completed = EXP_METADATA["EPOCHS_COMPLETED"]
            EPOCHS = EPOCHS - epochs_completed
            print(f"{epochs_completed} epochs have already been completed: the Fine-Tuning process will be ended with the remaining {EPOCHS} epochs")
            
            last_cp_path = f"{OUTPUT_DIR}/checkpoints/Test_{TEST_ID}_MLC_last_checkpoint.pth"
            last_cp = torch.load(last_cp_path)
            model.load_state_dict(last_cp['model_state_dict'])
         
        model.to(DEVICE)
        
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {pytorch_total_params}')
        
        trainer = Trainer(model=model, t_set=t_dl, v_set=v_dl, DEVICE=DEVICE, model_path=OUTPUT_DIR, 
                          history_path=OUTPUT_DIR, exp_metadata=EXP_METADATA, last_cp=last_cp)
        trainer()
        
        EXP_METADATA["FINE_TUNING_TIMESTAMP"] = str(datetime.now())
        save_metadata(EXP_METADATA, EXP_METADATA_PATH)
        print("...Model Fine-Tuning completed!\n")
        torch.cuda.empty_cache()
    
    if "MODEL_TESTING_TIMESTAMP" in EXP_METADATA:
        print(f"*** IN RELATION TO THE EXPERIMENT '{TEST_ID}', THE MODEL HAS ALREADY BEEN TESTED! ***\n")
    
    else:
        print("PHASE 3 -> MODEL TESTING...")
        for metric in ["loss", "accuracy"]: plot_metric(metric, OUTPUT_DIR, TEST_ID)
        
        cp_to_test = f"{OUTPUT_DIR}/checkpoints/Test_{TEST_ID}_MLC_val_best_model.pth"
        model = NN_Classifier(num_classes=len(CLASSES_DATA), mode='frozen', cp_path=MODEL_PATH)
        model.to(DEVICE)
        model.load_state_dict(torch.load(cp_to_test)['model_state_dict']) 
        model.eval()

        mean_, std_ = load_rgb_mean_std(f"{EXP_DIR}")
        dl = Train_Test_DataLoader(directory=f"{EXP_DIR}/test", classes=list(CLASSES_DATA.keys()), batch_size=1, img_crop_size=CROP_SIZE, weighted_sampling=False, phase='test', mean=mean_, std=std_, shuffle=True)
        produce_classification_reports(dl, DEVICE, model, OUTPUT_DIR, TEST_ID)
        print("...Testing reports are now available!\n")
    
        print("PHASE 4 -> DATA & METADATA HANDLING...")
        os.system(f"rm -r {OUTPUT_DIR}/checkpoints/Test_{TEST_ID}_MLC_last_checkpoint.pth")
        os.system(f"rm -r {EXP_DIR}/train")
        os.system(f"rm -r {EXP_DIR}/test")
        EXP_METADATA["MODEL_TESTING_TIMESTAMP"] = str(datetime.now())
        save_metadata(EXP_METADATA, EXP_METADATA_PATH)

        torch.cuda.empty_cache()
        print("*** END OF FINE-TUNING PROCESS ***\n")