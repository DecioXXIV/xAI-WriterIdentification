import torch, itertools
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

def process_test_set(dl, device, model):
    dataset, set_ = dl.load_data()
    
    labels, preds = list(), list()
    
    target_names = list(dataset.class_to_idx.keys())
    c_to_idx = dataset.class_to_idx
    idx_to_c = {c_to_idx[k]: k for k in list(c_to_idx.keys())}
    
    for data, target in tqdm(set_):
        if data.dim() == 4:
            labels.append(target.max().item())
            data, target = data.to(device), target.to(device)
            bs = data.size(0)
            
            with torch.no_grad():
                output = model(data)
                indices = output.max(dim=1)[1].cpu().numpy()
                max_index = np.argmax(np.bincount(indices))
                preds.append(max_index)
        
        if data.dim() == 5:
            labels.extend(list(target.numpy()))
            data, target = data.to(device), target.to(device)
            bs, ncrops, c, h, w = data.size()
            
            with torch.no_grad():
                output = model(data.view(-1, c, h, w))
                indices = output.max(dim=1)[1].cpu().numpy()
                indices_over_crops = indices.reshape(bs, ncrops)
                max_indices = list()
                
                for i in range(0, bs):
                    max_indices.append(np.argmax(np.bincount(indices_over_crops[i, :])))
                
                preds.extend(max_indices)
        
    return dataset, labels, preds, target_names, idx_to_c
    
def produce_confusion_matrix(labels, preds, target_names, idx_to_c, output_dir, test_id):
    label_class_names = np.array([idx_to_c[id_] for id_ in labels])
    pred_class_names = np.array([idx_to_c[id_] for id_ in preds])
    
    cm = confusion_matrix(label_class_names, pred_class_names, labels=target_names)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    
    plt.figure(figsize=(20, 20))
    cmap = plt.get_cmap('Blues')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max() / 1.5
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:0.2f}",
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\naccuracy = {accuracy:0.4f}; misclass = {misclass:0.4f}')
    plt.savefig(f'{output_dir}/Test_{test_id}_confusion_matrix_test.png')
    
def produce_classification_reports(dl, device, model, output_dir, test_id):
    dataset, labels, preds, target_names, idx_to_c = process_test_set(dl, device, model)
    
    produce_confusion_matrix(labels, preds, target_names, idx_to_c, output_dir, test_id)
        
    report = classification_report(labels, preds, target_names=target_names)
    with open(f'{output_dir}/Test_{test_id}_classification-report_test.txt', 'w') as f:
        f.write(report)
        
    with open(f"{output_dir}/class_to_idx.pkl", "wb") as f:
        pkl.dump(dataset.class_to_idx, f)