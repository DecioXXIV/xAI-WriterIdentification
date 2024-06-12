import torch
from utils import Classification_Model, Standard_DataLoader, load_rgb_mean_std, produce_classification_reports

cp_base = './cp/Test_3_TL_val_best_model.pth'
# cp = './cp/Test_0003_MLC_val_best_model.pth' # locale
cp = './output/checkpoints/Test_0003_MLC_val_best_model.pth' # remoto
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data_dir = 'D:/Post-doc/HI/split_42/' # locale
data_dir = './split_42/' # remoto
output_dir = './output/'
test_id = '0003'

model = Classification_Model(mode = 'frozen', cp_path = cp_base, num_classes = 4)

model = model.to(DEVICE)
model.load_state_dict(torch.load(cp)['model_state_dict'])
model.eval()

mean_, std_ = load_rgb_mean_std(f'{data_dir}train')

dl = Standard_DataLoader(f'{data_dir}test', 64, False, 'test', mean_, std_, True)

produce_classification_reports(dl, DEVICE, model, output_dir, test_id)