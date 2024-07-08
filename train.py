import argparse
import logging
import torchvision 
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from lora import LoRA_ViT_timm
from utils.dataloader_oai import kneeDataloader
from utils.dataloader_cxr_cn import cxrDataloader
from utils.dataloader_blood_cell import BloodDataloader
from utils.dataloader_cub import load_data_CLASS
from utils.dataloader_nih import nihDataloader
from utils.result import ResultCLS
from utils.utils import init, save
from utils.dataloader_har import load_data
import random
import numpy as np
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=6, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma
        
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
        
    def forward(self, preds, labels):
        assert preds.dim() == 2 and labels.dim() == 1
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def train(epoch, trainset):
    running_loss = 0.0
    this_lr = scheduler.get_last_lr()[0]
    net.train()
    for image, label in tqdm(trainset, ncols=60, desc="train", unit="b", leave=None):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast(enabled=True):
            pred = net.forward(image)
            loss = loss_func(pred, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss = running_loss + loss.item()
    scheduler.step()

    loss = running_loss / len(trainset)
    logging.info(f"\n\nEPOCH: {epoch}, LOSS : {loss:.3f}, LR: {this_lr:.2e}")
    return

@torch.no_grad()
def eval(epoch, testset, datatype='val'):
    result.init()
    net.eval()
    all_preds = []
    all_labels = []
    for image, label in tqdm(testset, ncols=60, desc=datatype, unit="b", leave=None):
        image, label = image.to(device), label.to(device)
        with autocast(enabled=True):
            pred = net.forward(image)
            _, preds = torch.max(pred, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            result.eval(label, pred)
    result.print(epoch, datatype)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    return cm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(42)
    scaler = GradScaler()
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", type=int, default=256)
    parser.add_argument("-fold", type=int, default=0)
    parser.add_argument("-data_path", type=str, default='aircraft')
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-epochs", type=int, default=200)
    parser.add_argument("-num_workers", type=int, default=8)
    parser.add_argument("-num_classes", "-nc", type=int, default=200)
    parser.add_argument("-train_type", "-tt", type=str, default="lora", help="lora, full, linear, adapter, lora_swin")
    parser.add_argument("-rank", "-r", type=int, default=4)
    parser.add_argument("-alpha", "-a", type=int, default=4)
    parser.add_argument("-vit", type=str, default="base")
    parser.add_argument("-data_size", type=float, default='1.0')
    parser.add_argument("-pretrained_path", type=str, default='/home/wsco/wyj/SelaFD/pretrain/vit_base_patch16_224.pth')
    cfg = parser.parse_args()
    ckpt_path = init(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logging.info(cfg)


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),                         
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if cfg.data_path == 'cub':
        trainset, valset, testset = load_data_CLASS(root='/home/wsco/wyj/HASH_Net/datasets/CUB_200_2011', batch_size=cfg.bs, num_workers=cfg.num_workers)
        cfg.num_classes = 200
    elif cfg.data_path == 'flowers':
        train_data = torchvision.datasets.Flowers102(root='./data', split='train',
                                        download=True, transform=train_transform)
        trainset = torch.utils.data.DataLoader(train_data, batch_size=cfg.bs,
                                                shuffle=True, num_workers=cfg.num_workers)

       
        test_data = torchvision.datasets.Flowers102(root='./data', split='test',
                                            download=True, transform=test_transform)
        testset = torch.utils.data.DataLoader(test_data, batch_size=cfg.bs,
                                                shuffle=False, num_workers=cfg.num_workers)
        valset=testset
        cfg.num_classes = 102
    elif cfg.data_path == 'aircraft':
        train_data = torchvision.datasets.FGVCAircraft(root='./data', split='trainval',
                                        download=True, transform=train_transform)
        trainset = torch.utils.data.DataLoader(train_data, batch_size=cfg.bs,
                                                shuffle=True, num_workers=cfg.num_workers)

       
        test_data = torchvision.datasets.FGVCAircraft(root='./data', split='test',
                                            download=True, transform=test_transform)
        testset = torch.utils.data.DataLoader(test_data, batch_size=cfg.bs,
                                                shuffle=False, num_workers=cfg.num_workers)
        valset=testset
        cfg.num_classes = 100
    elif cfg.data_path == 'food':
        train_data = torchvision.datasets.Food101(root='./data', split='train',
                                        download=True, transform=train_transform)
        trainset = torch.utils.data.DataLoader(train_data, batch_size=cfg.bs,
                                                shuffle=True, num_workers=cfg.num_workers)

       
        test_data = torchvision.datasets.Food101(root='./data', split='test',
                                            download=True, transform=test_transform)
        testset = torch.utils.data.DataLoader(test_data, batch_size=cfg.bs,
                                                shuffle=False, num_workers=cfg.num_workers)
        valset=testset
        cfg.num_classes = 101
    elif cfg.data_path == 'har':
        trainset, valset, testset = load_data(root='/media/wsco/Dataset_848/', batch_size=cfg.bs, num_workers=cfg.num_workers)
        cfg.num_classes = 6

    model = timm.create_model('vit_base_patch16_224',pretrained=True)
    if cfg.train_type == "lora":
        lora_model = LoRA_ViT_timm(r=cfg.rank, alpha=cfg.alpha, num_classes=cfg.num_classes, pretrained_path=cfg.pretrained_path)
        num_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"trainable parameters: {num_params/2**20:.3f}M")
        print("tuned percent:%.3f%%" % (num_params / total_params * 100))
        net = lora_model.to(device)
    elif cfg.train_type == "full":
        model.reset_classifier(cfg.num_classes)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable parameters: {num_params / 2**20:.3f}M")
        net = model.to(device)
    elif cfg.train_type == "linear":
        model.reset_classifier(cfg.num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        num_params = sum(p.numel() for p in model.head.parameters())
        print(f"trainable parameters: {num_params / 2**20:.3f}M")
        net = model.to(device)
    elif cfg.train_type == 'resnet50':
        infeature = model.fc.in_features
        model.fc = nn.Linear(infeature, cfg.num_classes)
        num_params = sum(p.numel() for p in model.fc.parameters())
        print(f"trainable parameters: {num_params / 2**20:.3f}M")
        net = model.to(device)
    else:
        print("Wrong training type")
        exit()

    

    
    
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    # loss_func = FocalLoss(gamma=2, alpha=[2,0.5,2,1,0.5,0.5])
    # loss_func = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, cfg.epochs, 1e-6)
    result = ResultCLS(cfg.num_classes)

    for epoch in range(1, cfg.epochs + 1):
        train(epoch, trainset)
        if epoch % 1 == 0:
            cm = eval(epoch, valset, datatype='val')
            if result.best_epoch == result.epoch:
                logging.info(f'EPOCH: {(result.best_epoch):3},>>>>>>>>>>>>>>>>>>>>>>>>save model')
                torch.save(net.state_dict(), ckpt_path)
                if cfg.train_type == 'lora':
                    net.save_lora_parameters(ckpt_path.replace(".pt", ".safetensors"))
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
                class_labels = ['drink_water', 'fall', 'pick_up', 'sit_down', 'stand_up', 'walk']

                # Plot the confusion matrix
                plt.figure(figsize=(10, 10))
                sns.heatmap(cm_with_accuracies, annot=annot, fmt='', cmap='Blues', cbar=False,
                            xticklabels=class_labels, yticklabels=class_labels)
                plt.xlabel('Predicted Labels')
                plt.xlabel('Predicted Labels')
                plt.ylabel('True Labels')
                plt.title('Confusion Matrix with Classification Accuracy and Counts')
                plt.savefig('img_res/loada_confusion_matrix.png')
            # eval(epoch, testset, datatype='test')
            logging.info(f"BEST VAL: {result.best_val_result:.4f}, EPOCH: {(result.best_epoch):3}")
            # logging.info(result.test_mls_auc)
