import os, torch
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

from utils import load_metadata, save_metadata, get_model_base_checkpoint

from classifiers.utils.dataloader_utils import Train_DataLoader, Test_DataLoader, load_rgb_mean_std
from classifiers.utils.training_utils import Trainer, plot_metric
from classifiers.utils.testing_utils import produce_classification_reports
from classifiers.utils.fine_tune_utils import create_directories, extract_crops_parallel, retrieve_augmentation_crops, load_model

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
CLASSIFIERS_ROOT = "./classifiers"

def get_args():
    parser = ArgumentParser(description="Setup of Fine-tuning hyperparameters")
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-crop_size", type=int, required=True)
    parser.add_argument("-opt", type=str, required=True)
    parser.add_argument("-lr", type=float, required=True)
    parser.add_argument("-train_replicas", type=int, required=True)
    parser.add_argument("-train_dl_mf", type=int, default=1)
    parser.add_argument("-random_seed", type=int, default=None)
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-ft_mode", type=str, default="frozen", choices=["frozen", "full"])
    
    return parser.parse_args()

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
    
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]
    DATASET = EXP_METADATA["DATASET"]
    CLASSES_DATA = EXP_METADATA["CLASSES"]
    
    CROP_SIZE = args.crop_size
    OPT = args.opt
    LR = args.lr
    TRAIN_REPLICAS = args.train_replicas
    TRAIN_DL_MF = args.train_dl_mf
    RANDOM_SEED = args.random_seed if args.random_seed is not None else np.random.randint(0, 10e6)
    TEST_N_CROPS = 250
    EPOCHS = args.epochs
    FT_MODE = args.ft_mode
    
    SOURCE_DATA_DIR = f"{DATASET_ROOT}/{DATASET}/processed"
    EXP_DIR = f"{CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/tests/{TEST_ID}"
    OUTPUT_DIR = f"{CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/tests/{TEST_ID}/output"
    CP_BASE = get_model_base_checkpoint(MODEL_TYPE)
    
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
            "crop_size": CROP_SIZE,
            "train_replicas": TRAIN_REPLICAS,
            "train_dl_mf": TRAIN_DL_MF,
            "random_seed": RANDOM_SEED,
            "total_epochs": EPOCHS
        }
        save_metadata(EXP_METADATA, EXP_METADATA_PATH)
    
        os.makedirs(f"{CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/tests", exist_ok=True)
        create_directories(EXP_DIR, CLASSES_DATA.keys())
    
        if "FT_DATASET_PREP_TIMESTAMP" not in EXP_METADATA:
            print("PHASE 1 -> DATASET CREATION...")
            for c, c_type in CLASSES_DATA.items():
                class_source, class_dest = None, None
                if c_type == "base": class_source = f"{SOURCE_DATA_DIR}/{c}"
                else: class_source = f"{SOURCE_DATA_DIR}/{c}-{BASE_ID}_NN_{c_type}"
        
                extract_crops_parallel(DATASET, EXP_DIR, class_source, c, TRAIN_REPLICAS, CROP_SIZE, TRAIN_DL_MF, TEST_N_CROPS, RANDOM_SEED)
                
                if "augmented" or "refined" in TEST_ID:
                    BASE_ID, _ = TEST_ID.split(':')
                    retrieve_augmentation_crops(TEST_ID, BASE_ID, MODEL_TYPE, c)
            
                class_n_train_crops = len(os.listdir(f"{EXP_DIR}/train/{c}"))
                print(f"Class {c} -> Train Instances ({CROP_SIZE}x{CROP_SIZE}-sized Crops): {class_n_train_crops}")
                    
            EXP_METADATA["FT_DATASET_PREP_TIMESTAMP"] = str(datetime.now())
            save_metadata(EXP_METADATA, EXP_METADATA_PATH)
            print("...Dataset Created!\n")
    
        else: print("Skipping PHASE 1 (Dataset Creation): it's already available!\n")
    
        mean_, std_ = load_rgb_mean_std(f"{EXP_DIR}/train")
        os.rename(f"{EXP_DIR}/train/rgb_train_stats.pkl", f"{EXP_DIR}/rgb_train_stats.pkl")
    
        print("PHASE 2 -> MODEL FINE-TUNING...")    
        os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)

        # For Experiment reproducibility
        torch.manual_seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
                
        train_ds = Train_DataLoader(directory=f"{EXP_DIR}/train", classes=list(CLASSES_DATA.keys()), batch_size=96, img_crop_size=CROP_SIZE, mult_factor=TRAIN_DL_MF, weighted_sampling=True, mean=mean_, std=std_, shuffle=True)
        val_ds = Test_DataLoader(directory=f"{EXP_DIR}/val", classes=list(CLASSES_DATA.keys()), batch_size=96, img_crop_size=CROP_SIZE, mean=mean_, std=std_)
        tds, t_dl = train_ds.load_data()
        vds, v_dl = val_ds.load_data()

        model, last_cp = load_model(MODEL_TYPE, len(CLASSES_DATA), FT_MODE, CP_BASE, "train", TEST_ID, EXP_METADATA, DEVICE)
        if "EPOCHS_COMPLETED" in EXP_METADATA:
            epochs_completed = EXP_METADATA["EPOCHS_COMPLETED"]
            EPOCHS = EPOCHS - epochs_completed
        
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {pytorch_total_params}')
        
        trainer = Trainer(model=model, t_set=t_dl, v_set=v_dl, DEVICE=DEVICE, model_path=OUTPUT_DIR, history_path=OUTPUT_DIR, exp_metadata=EXP_METADATA, last_cp=last_cp)
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
        model, _ = load_model(MODEL_TYPE, len(CLASSES_DATA), "frozen", CP_BASE, "test", TEST_ID, EXP_METADATA, DEVICE)

        mean_, std_ = load_rgb_mean_std(f"{EXP_DIR}")
        dl = Test_DataLoader(directory=f"{EXP_DIR}/test", classes=list(CLASSES_DATA.keys()), batch_size=TEST_N_CROPS, img_crop_size=CROP_SIZE, mean=mean_, std=std_)
        produce_classification_reports(dl, DEVICE, model, OUTPUT_DIR, TEST_ID)
        print("...Testing reports are now available!\n")
    
        print("PHASE 4 -> DATA & METADATA HANDLING...")
        os.system(f"rm -r {OUTPUT_DIR}/checkpoints/Test_{TEST_ID}_MLC_last_checkpoint.pth")
        os.system(f"rm -r {EXP_DIR}/train")
        os.system(f"rm -r {EXP_DIR}/val")
        os.system(f"rm -r {EXP_DIR}/test")
        EXP_METADATA["MODEL_TESTING_TIMESTAMP"] = str(datetime.now())
        save_metadata(EXP_METADATA, EXP_METADATA_PATH)

        torch.cuda.empty_cache()
        print("*** END OF FINE-TUNING PROCESS ***\n")