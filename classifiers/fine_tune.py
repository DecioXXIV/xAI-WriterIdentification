import os, torch
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

from utils import str2bool, load_metadata, save_metadata, get_model_base_checkpoint, get_logger

from classifiers import SCHEDULERS, FT_MODES
from classifiers.utils.dataloader_utils import Train_DataLoader, Test_DataLoader, load_rgb_mean_std
from classifiers.utils.training_utils import Trainer, plot_metric, plot_learning_rates
from classifiers.utils.fine_tune_utils import create_directories, extract_crops_parallel, load_model, test_fine_tuned_model, retrieve_augmentation_crops

LOG_ROOT = "./log"
DATASET_ROOT = "./datasets"
CLASSIFIERS_ROOT = "./classifiers"

def get_args():
    parser = ArgumentParser(description="Setup of Fine-tuning hyperparameters")
    parser.add_argument("-test_id", type=str, required=True)
    parser.add_argument("-crop_size", type=int, required=True)
    parser.add_argument("-batch_size", type=int, required=True)
    parser.add_argument("-opt", type=str, required=True)
    parser.add_argument("-lr", type=float, required=True)
    parser.add_argument("-scheduler", type=str, default=None, choices=SCHEDULERS)
    parser.add_argument("-early_stopping", type=str2bool, default=True)
    parser.add_argument("-train_replicas", type=int, required=True)
    parser.add_argument("-random_seed", type=int, default=None)
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-ft_mode", type=str, default="frozen", choices=FT_MODES)
    parser.add_argument("-keep_crops", type=str2bool, default=False)
    
    return parser.parse_args()

### #### ###
### MAIN ###
### #### ###
if __name__ == '__main__':
    args = get_args()
    TEST_ID = args.test_id
    logger = get_logger(TEST_ID)
    
    EXP_METADATA_PATH = f"{LOG_ROOT}/{TEST_ID}-metadata.json"
    try: EXP_METADATA = load_metadata(EXP_METADATA_PATH, logger)
    except Exception as e:
        logger.error(f"Failed to load Metadata for experiment {TEST_ID}")
        logger.error(f"Details: {e}")
        exit(1)
    
    MODEL_TYPE = EXP_METADATA["MODEL_TYPE"]
    DATASET = EXP_METADATA["DATASET"]
    CLASSES_DATA = EXP_METADATA["CLASSES"]
    
    CROP_SIZE = args.crop_size
    BATCH_SIZE = args.batch_size
    OPT = args.opt
    LR = args.lr
    SCHEDULER = args.scheduler
    EARLY_STOPPING = args.early_stopping
    TRAIN_REPLICAS = args.train_replicas
    RANDOM_SEED = args.random_seed if args.random_seed is not None else np.random.randint(0, 10e6)
    EPOCHS = args.epochs
    FT_MODE = args.ft_mode
    KEEP_CROPS = args.keep_crops
    
    SOURCE_DATA_DIR = f"{DATASET_ROOT}/{DATASET}/processed"
    EXP_DIR = f"{CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/tests/{TEST_ID}"
    OUTPUT_DIR = f"{CLASSIFIERS_ROOT}/classifier_{MODEL_TYPE}/tests/{TEST_ID}/output"
    CP_BASE = get_model_base_checkpoint(MODEL_TYPE)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if "FINE_TUNING_TIMESTAMP" in EXP_METADATA: 
        logger.warning(f"*** IN RELATION TO THE EXPERIMENT '{TEST_ID}', THE MODEL HAS ALREADY BEEN FINE-TUNED! ***")
    else:
        logger.info(f"*** Experiment: {TEST_ID} -> BEGINNING OF FINE-TUNING PROCESS ***")
        
        BASE_ID, RET_ID = None, None
        try: BASE_ID, RET_ID = TEST_ID.split(':')
        except: BASE_ID = TEST_ID
        
        if "FINE_TUNING_HP" not in EXP_METADATA:
            EXP_METADATA["FINE_TUNING_HP"] = {"batch_size": BATCH_SIZE, "optimizer": OPT, "learning_rate": LR, "scheduler": SCHEDULER, 
                                              "early_stopping": EARLY_STOPPING, "crop_size": CROP_SIZE,  "train_replicas": TRAIN_REPLICAS, 
                                              "random_seed": RANDOM_SEED, "total_epochs": EPOCHS, "ft_mode": FT_MODE}
            save_metadata(EXP_METADATA, EXP_METADATA_PATH)
        
        create_directories(EXP_DIR, CLASSES_DATA.keys())
        
        if "FT_DATASET_PREP_TIMESTAMP" not in EXP_METADATA:
            logger.info("PHASE 1 -> DATASET CREATION...")
            
            for c, c_type in CLASSES_DATA.items():
                class_source = None
                
                if "ret" in TEST_ID:
                    logger.info(f"'{TEST_ID}' is a 'ret' experiment: training Crops will be extracted from the 'masked' instances!")
                    class_source = f"{SOURCE_DATA_DIR}/{c}-{BASE_ID}_{MODEL_TYPE}_{c_type}"
                    
                else: class_source = f"{SOURCE_DATA_DIR}/{c}"
                
                n_crops_per_train_instance, n_crops_per_test_instance = extract_crops_parallel(DATASET, EXP_DIR, class_source, c, TRAIN_REPLICAS, CROP_SIZE, train_dl_mf=1)
                
                EXP_METADATA["FINE_TUNING_HP"]["n_crops_per_train_instance"] = n_crops_per_train_instance
                EXP_METADATA["FINE_TUNING_HP"]["n_crops_per_test_instance"] = n_crops_per_test_instance
                save_metadata(EXP_METADATA, EXP_METADATA_PATH)  
            
            logger.info("Base training Crops Extracted!")
            
            if "augmented" in TEST_ID:
                logger.info(f"'{TEST_ID}' is an 'augmented' experiment: XAI-driven augmented Crops have been added to the Dataset!")
                for c in CLASSES_DATA.keys(): 
                    retrieve_augmentation_crops(TEST_ID, MODEL_TYPE, c)
                
            mean_, std_ = load_rgb_mean_std(f"{EXP_DIR}/train_pre_aug", logger)
            os.rename(f"{EXP_DIR}/train_pre_aug/rgb_train_stats.pkl", f"{EXP_DIR}/rgb_train_stats.pkl")
            
            EXP_METADATA["FINE_TUNING_HP"]["mean"] = mean_
            EXP_METADATA["FINE_TUNING_HP"]["std"] = std_
            EXP_METADATA["FT_DATASET_PREP_TIMESTAMP"] = str(datetime.now())
            save_metadata(EXP_METADATA, EXP_METADATA_PATH)
            logger.info("...Dataset Created!\n")
        
        else:
            logger.info("Skipping PHASE 1 (Dataset Creation): it's already available!\n") 
        
        logger.info("PHASE 2 -> MODEL FINE-TUNING...")
        os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)
        
        # Uncomment for Experiment Reproducibility
        # torch.manual_seed(RANDOM_SEED)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        mean_, std_ = load_rgb_mean_std(f"{EXP_DIR}", logger)
        train_ds = Train_DataLoader(model_type=MODEL_TYPE, directory=f"{EXP_DIR}/train_pre_aug", classes=list(CLASSES_DATA.keys()), batch_size=BATCH_SIZE, img_crop_size=CROP_SIZE, mean=mean_, std=std_, shuffle=True)
        val_ds = Test_DataLoader(model_type=MODEL_TYPE, directory=f"{EXP_DIR}/val", classes=list(CLASSES_DATA.keys()), batch_size=BATCH_SIZE, img_crop_size=CROP_SIZE, mean=mean_, std=std_)
        _, t_dl = train_ds.load_data()
        _, v_dl = val_ds.load_data()
        
        model, last_cp = load_model(MODEL_TYPE, len(CLASSES_DATA), FT_MODE, CP_BASE, "train", TEST_ID, EXP_METADATA, DEVICE, logger)
        if "EPOCHS_COMPLETED" in EXP_METADATA:
            epochs_completed = EXP_METADATA["EPOCHS_COMPLETED"]
            EPOCHS -= epochs_completed
        
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Number of trainable parameters: {pytorch_total_params}')
        
        trainer = Trainer(model=model, t_set=t_dl, v_set=v_dl, DEVICE=DEVICE, model_path=OUTPUT_DIR, history_path=OUTPUT_DIR, exp_metadata=EXP_METADATA, use_early_stopping=EARLY_STOPPING, last_cp=last_cp, logger=logger)
        trainer()
        
        EXP_METADATA["FINE_TUNING_TIMESTAMP"] = str(datetime.now())
        save_metadata(EXP_METADATA, EXP_METADATA_PATH)
        for metric in ["loss", "accuracy"]: plot_metric(metric, OUTPUT_DIR, TEST_ID)
        plot_learning_rates(OUTPUT_DIR, TEST_ID)
        torch.cuda.empty_cache()
        logger.info("...Model Fine-Tuning completed!\n")

    if "MODEL_TESTING_TIMESTAMP" in EXP_METADATA:
        logger.warning(f"*** IN RELATION TO THE EXPERIMENT '{TEST_ID}', THE MODEL HAS ALREADY BEEN TESTED! ***")

    else:
        logger.info("PHASE 3 -> MODEL TESTING...")
        mean_, std_ = EXP_METADATA["FINE_TUNING_HP"]["mean"], EXP_METADATA["FINE_TUNING_HP"]["std"]
        test_fine_tuned_model(TEST_ID, EXP_METADATA, MODEL_TYPE, CROP_SIZE, OUTPUT_DIR, CLASSES_DATA, CP_BASE, mean_, std_, logger)
        logger.info("...Testing reports are now available!\n")
     
        logger.info("PHASE 4 -> DATA & METADATA HANDLING...")
        if not KEEP_CROPS:
            os.system(f"rm -r {EXP_DIR}/train_pre_aug")
            os.system(f"rm -r {EXP_DIR}/train")
            os.system(f"rm -r {EXP_DIR}/val")
            os.system(f"rm -r {EXP_DIR}/test")
        os.system(f"rm -r {OUTPUT_DIR}/checkpoints/Test_{TEST_ID}_MLC_last_checkpoint.pth")
        EXP_METADATA["MODEL_TESTING_TIMESTAMP"] = str(datetime.now())
        save_metadata(EXP_METADATA, EXP_METADATA_PATH)
        
        torch.cuda.empty_cache()
        logger.info(f"*** Experiment: {TEST_ID} -> END OF FINE-TUNING PROCESS ***\n")