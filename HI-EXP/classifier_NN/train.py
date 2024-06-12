import torch
from utils import Classification_Model, Standard_DataLoader, Trainer, load_rgb_mean_std

cp = './cp/Test_3_TL_val_best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = 'D:/Post-doc/HI/split_42/'
output_dir = './output/'

model = Classification_Model(mode = 'frozen', cp_path = cp, num_classes = 4)

model = model.to(DEVICE)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Number of trainable parameters: {pytorch_total_params}')

mean_, std_ = load_rgb_mean_std(f'{data_dir}train')
t_ds = Standard_DataLoader(f'{data_dir}train', 64, True, 'train', mean_, std_, True)
v_ds = Standard_DataLoader(f'{data_dir}val', 64, False, 'val', mean_, std_, True)
tds, t_dl = t_ds.load_data()
vds, v_dl = v_ds.load_data()

trainer = Trainer(model, t_dl, v_dl, DEVICE, 'adam', 0.01, output_dir, output_dir, '0003', 100)
trainer()