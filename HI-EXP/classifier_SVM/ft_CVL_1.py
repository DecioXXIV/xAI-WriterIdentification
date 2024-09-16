import os, torch, pickle
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from model import SVM_Classifier, load_tail
from utils import Standard_DataLoader, Trainer, produce_classification_reports, load_rgb_mean_std

def get_args():
    parser = ArgumentParser(description="Hyperparameters for the Fine-Tuning on Camera Dataset", add_help=True)
    parser.add_argument("-test_id", type=str)
    parser.add_argument("-crop_size", type=int)
    # parser.add_argument("-train_instances", type=int)
    parser.add_argument("-train_replicas", type=int)
    parser.add_argument("-random_seed", type=int)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    TEST_ID = args.test_id
    CROP_SIZE = args.crop_size
    # TRAIN_INSTANCES = args.train_instances
    TRAIN_REPLICAS = args.train_replicas
    RANDOM_SEED = args.random_seed

    CWD = os.getcwd()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = CWD + "/cp/Test_3_TL_val_best_model.pth"
    DATASET_DIR = CWD + f"/tests/{TEST_ID}"
    OUTPUT_DIR = CWD + f"/tests/{TEST_ID}/output"
    SOURCE_DATA_DIR = CWD + "/../../datasets/CVL_set1PNG/final_pages"

    print("PHASE 1 -> DATASET CREATION...")
    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)
        os.mkdir(DATASET_DIR + "/train")
        os.mkdir(DATASET_DIR + "/val")
        os.mkdir(DATASET_DIR + "/test")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    
    classes = os.listdir(SOURCE_DATA_DIR)
    # Ogni classe -> 84 istanze -> 60 train, 24 test

    for c in classes:
        if not os.path.exists(DATASET_DIR + "/train/" + c):
            os.mkdir(DATASET_DIR + "/train/" + c)
        if not os.path.exists(DATASET_DIR + "/val/" + c):
            os.mkdir(DATASET_DIR + "/val/" + c)
        if not os.path.exists(DATASET_DIR + "/test/" + c):
            os.mkdir(DATASET_DIR + "/test/" + c)
    
    for c in classes:
        all_instances = os.listdir(SOURCE_DATA_DIR + "/" + c)
        train, test = list(), list()

        for instance in all_instances:
            if ("-3" not in instance) and ("-7" not in instance): train.append(instance)
            else: test.append(instance)
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(test)
        val = np.random.choice(test, size=28, replace=False)

        for instance in train:
            for i in range(0, TRAIN_REPLICAS):
                os.system(f"cp {SOURCE_DATA_DIR}/{c}/{instance} {DATASET_DIR}/train/{c}/{instance[:-4]}_cp{i+1}.jpg")
        
        for instance in val:
            for i in range(0, TRAIN_REPLICAS):
                os.system(f"cp {SOURCE_DATA_DIR}/{c}/{instance} {DATASET_DIR}/val/{c}/{instance[:-4]}_cp{i+1}.jpg")
        
        for instance in test:
            os.system(f"cp {SOURCE_DATA_DIR}/{c}/{instance} {DATASET_DIR}/test/{c}/{instance[:-4]}.jpg")
    
    for c in classes:
        print(f"Class {c} -> Train Instances: {len(os.listdir(DATASET_DIR + '/train/' + c))}")
    print("...Dataset Created!\n")
    
    print("PHASE 2 -> MODEL TRAINING...")
    model = SVM_Classifier(mode='frozen', cp_path=MODEL_PATH)
    model.base_model.to(DEVICE)
    model.base_model.eval()

    mean_, std_ = load_rgb_mean_std(DATASET_DIR + "/train")
    train_ds = Standard_DataLoader(directory=f"{DATASET_DIR}/train", batch_size=64, img_crop_size=CROP_SIZE, weighted_sampling=True, phase='train', mean=mean_, std=std_, shuffle=True)
    tds, t_dl = train_ds.load_data()

    # MODEL TRAINING
    os.mkdir(f"{OUTPUT_DIR}/checkpoints")
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(model=model, t_set=t_dl, DEVICE=DEVICE, model_path=OUTPUT_DIR, test_ID=TEST_ID)
    trainer()
    print("...Model Trained!\n")

    # TESTING
    print("PHASE 3 -> MODEL TESTING...")
    torch.cuda.empty_cache()
    cp_base = f"./cp/Test_3_TL_val_best_model.pth"
    cp = f"{OUTPUT_DIR}/checkpoints/Test_{TEST_ID}_MLC_val_best_model.pth"

    model = SVM_Classifier(mode='frozen', cp_path=MODEL_PATH)
    model.base_model.to(DEVICE)
    model.base_model.eval()
    model.tail = load_tail(CWD + f"/tests/{TEST_ID}/output/checkpoints/svm_tail.pkl")

    test_ds = Standard_DataLoader(directory=f"{DATASET_DIR}/test", batch_size=64, img_crop_size=CROP_SIZE, weighted_sampling=False, phase='test', mean=mean_, std=std_, shuffle=False)
    produce_classification_reports(test_ds, DEVICE, model, OUTPUT_DIR, TEST_ID)
    print("...Testing reports are now available!\n")