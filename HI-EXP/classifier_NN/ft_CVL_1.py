import os, torch, pickle
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from model import NN_Classifier
from utils import Standard_DataLoader, Trainer, produce_classification_reports, load_rgb_mean_std

def get_args():
    parser = ArgumentParser(description="Hyperparameters for the Fine-Tuning on Camera Dataset", add_help=True)
    parser.add_argument("-test_id", type=str)
    parser.add_argument("-crop_size", type=int)
    parser.add_argument("-opt", type=str)
    parser.add_argument("-lr", type=float)
    # parser.add_argument("-train_instances", type=int)
    parser.add_argument("-train_replicas", type=int)
    parser.add_argument("-random_seed", type=int)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    TEST_ID = args.test_id
    CROP_SIZE = args.crop_size
    OPT = args.opt
    LR = args.lr
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
    num_classes = len(os.listdir(DATASET_DIR + "/test"))
    model = NN_Classifier(num_classes=num_classes, mode='frozen', cp_path=MODEL_PATH)
    model = model.to(DEVICE)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of classes: {num_classes}")
    print(f'Number of trainable parameters: {pytorch_total_params}')

    mean_, std_ = load_rgb_mean_std(f"{DATASET_DIR}/train")
    # mean_, std_ = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    train_ds = Standard_DataLoader(directory=f"{DATASET_DIR}/train", batch_size=128, img_crop_size=CROP_SIZE, weighted_sampling=True, phase='train', mean=mean_, std=std_, shuffle=True)
    val_ds = Standard_DataLoader(directory=f"{DATASET_DIR}/val", batch_size=128, img_crop_size=CROP_SIZE, weighted_sampling=False, phase='val', mean=mean_, std=std_, shuffle=False)
    tds, t_dl = train_ds.load_data()
    vds, v_dl = val_ds.load_data()

    os.mkdir(f"{OUTPUT_DIR}/checkpoints")
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(model=model, t_set=t_dl, v_set=v_dl, DEVICE=DEVICE, optim_type=OPT, lr_=LR, 
                  model_path=OUTPUT_DIR, history_path=OUTPUT_DIR, test_ID=TEST_ID, num_epochs=100)
    trainer()

    losses = {'train': [], 'val': []}
    accs = {'train': [], 'val': []}

    for loss in list(losses.keys()):
        with open(f'{OUTPUT_DIR}/Test_{TEST_ID}_MLC_{loss}_losses.pkl', 'rb') as f:
            losses[loss] = pickle.load(f)
    
    with open(f'{OUTPUT_DIR}/Test_{TEST_ID}_MLC_losses.txt', 'w') as f:
        f.write('The optimal value of loss for the training set is: {:01.3f}\n'.format(np.min(losses['train'])))
        f.write('The optimal value of loss for the validation set is: {:01.3f}\n'.format(np.min(losses['val'])))
        best_epoch_train = np.where(np.array(losses['train']) == min(losses['train']))[0][0] + 1
        best_epoch_val = np.where(np.array(losses['val']) == min(losses['val']))[0][0] + 1
        f.write(f"Epoch corresponding to the optimal value of the training loss: {best_epoch_train}\\{len(losses['train'])}\n")
        f.write(f"Epoch corresponding to the optimal value of the validation loss: {best_epoch_val}\\{len(losses['val'])}\n")
    
    plt.plot(losses['train'])
    plt.plot(losses['val'])
    plt.title('Model loss')
    plt.ylabel('Loss [-]')
    plt.xlabel('Epoch [-]')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.savefig(f'{OUTPUT_DIR}/Test_{TEST_ID}_MLC_losses.png')
    plt.close()

    for acc in list(accs.keys()):
        with open(f'{OUTPUT_DIR}/Test_{TEST_ID}_MLC_{acc}_accuracy.pkl', 'rb') as f:
            accs[acc] = pickle.load(f)
    
    with open(f'{OUTPUT_DIR}/Test_{TEST_ID}_MLC_accuracy.txt', 'w') as f:
        f.write('The optimal value of accuracy for the training set is: {:01.3f}\n'.format(np.max(accs['train'])))
        f.write('The optimal value of accuracy for the validation set is: {:01.3f}\n'.format(np.max(accs['val'])))
        best_epoch_train = np.where(np.array(accs['train']) == max(accs['train']))[0][0] + 1
        best_epoch = np.where(np.array(accs['val']) == max(accs['val']))[0][0] + 1
        f.write(f"Epoch corresponding to the optimal value of the training accuracy: {best_epoch_train}\\{len(accs['train'])}\n")
        f.write(f"Epoch corresponding to the optimal value of the validation accuracy: {best_epoch}\\{len(accs['val'])}\n")
    
    plt.plot(accs['train'])
    plt.plot(accs['val'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy [-]')
    plt.xlabel('Epoch [-]')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.savefig(f'{OUTPUT_DIR}/Test_{TEST_ID}_MLC_accuracy.png')
    plt.close()
    print("...Model Trained!\n")

    print("PHASE 3 -> MODEL TESTING...")  
    torch.cuda.empty_cache()
    cp_base = f"./cp/Test_3_TL_val_best_model.pth"
    cp = f"{OUTPUT_DIR}/checkpoints/Test_{TEST_ID}_MLC_val_best_model.pth"

    model_to_test = NN_Classifier(num_classes=num_classes, mode='frozen', cp_path=cp_base)
    model_to_test = model_to_test.to(DEVICE)
    model_to_test.load_state_dict(torch.load(cp)['model_state_dict'])
    model.eval()

    dl = Standard_DataLoader(f"{DATASET_DIR}/test", 128, CROP_SIZE, False, 'test', mean_, std_, True)
    produce_classification_reports(dl, DEVICE, model, OUTPUT_DIR, TEST_ID)
    print("...Testing reports are now available!\n")