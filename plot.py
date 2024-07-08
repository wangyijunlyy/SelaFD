import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import timm
from lora import LoRA_ViT_timm
import argparse
from utils.dataloader_har import load_data
from utils.result import ResultCLS
from torch.cuda.amp.autocast_mode import autocast
# from vit_timm import vit_base_patch16_224
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# set_seed(42)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-bs", type=int, default=128)
parser.add_argument("-fold", type=int, default=0)
parser.add_argument("-data_path", type=str, default='har')
parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("-epochs", type=int, default=200)
parser.add_argument("-num_workers", type=int, default=8)
parser.add_argument("-num_classes", "-nc", type=int, default=6)
parser.add_argument("-train_type", "-tt", type=str, default="lora", help="lora, full, linear, adapter, lora_swin")
parser.add_argument("-rank", "-r", type=int, default=4)
parser.add_argument("-alpha", "-a", type=int, default=4)
parser.add_argument("-vit", type=str, default="base")
parser.add_argument("-data_size", type=float, default='1.0')
parser.add_argument("-pretrained_path", type=str, default='/home/wsco/wyj/SelaFD/results/4_lora_har_20240702_162914.pt')
cfg = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = LoRA_ViT_timm(r=cfg.rank, alpha=cfg.alpha, num_classes=cfg.num_classes)
model.load_state_dict(torch.load(cfg.pretrained_path))
net = model.load_lora_parameters(cfg.pretrained_path.replace('pt','safetensors'))
net = model.to(device)
print(net)


# Data loading
_, _, test_dataloader = load_data(root='/media/wsco/Dataset_848/', batch_size=cfg.bs, num_workers=cfg.num_workers)

# Model evaluation
net.eval()
all_preds = []
all_labels = []
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print(cm)
# Calculate accuracy for each class
accuracies = cm.diagonal() / cm.sum(axis=1)
cm_with_accuracies = cm.astype(float)

# Annotate the confusion matrix with counts and accuracy
for i in range(cm.shape[0]):
    cm_with_accuracies[i, i] = cm[i, i]

# Create annotations for the heatmap
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        if i == j:
            annot[i, j] = f'{c}\n{accuracies[i]*100:.2f}%'
        else:
            annot[i, j] = f'{c}'
# Define class labels
class_labels = ['drinking', 'falling', 'picking up', 'sitting', 'standing', 'walking']

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm_with_accuracies, annot=annot, fmt='', cmap='Oranges', cbar=False,
            xticklabels=class_labels, yticklabels=class_labels,annot_kws={"size": 14, "weight": 'bold'})
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.savefig('img_res/confusion_matrix_95.76.png', bbox_inches='tight', pad_inches=0)

