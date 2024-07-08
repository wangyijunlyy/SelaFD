import torch
import numpy as np


import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image, ImageFile
from torchvision import transforms

def train_transform():
    """
    Training images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(224),                         
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


def test_transform():
    """
    Test images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        # transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

def load_data(root,batch_size,num_workers):

    HAR.init(root)

    test_dataset = HAR(root, 'test', test_transform())
    train_dataset = HAR(root, 'train', train_transform())
    
    print(len(test_dataset))
    print(len(train_dataset))

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    

    return train_dataloader,test_dataloader,test_dataloader

def read_csv(csv_file):
    df = pd.read_csv(csv_file)
    image_paths = df['image_path'].to_numpy()
    labels = df['label'].to_numpy()
    return image_paths, labels


class HAR(Dataset):

    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader

        if mode == 'train':
            self.data = HAR.TRAIN_DATA
            self.targets = HAR.TRAIN_TARGETS
        elif mode == 'test':
            self.data = HAR.TEST_DATA
            self.targets = HAR.TEST_TARGETS
        
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')



    @staticmethod
    def init(root):
        train_csv_file = os.path.join(root,"train1_4.csv")
        test_csv_file = os.path.join(root,"test1_4.csv")
        # data_file = os.path.join(root,'train_test.csv')
        # data_paths,data_labels = read_csv(data_file)
        
        # 获取训练集和测试集数据
        train_image_paths, train_labels = read_csv(train_csv_file)
        test_image_paths, test_labels = read_csv(test_csv_file)
        # train_image_paths = data_paths[train_indices]
        # train_labels = data_labels[train_indices]
        # test_image_paths = data_paths[val_indices]
        # test_labels = data_labels[val_indices]


        # HAR.DATA = data_paths
        # HAR.TARGETS = data_labels
        # Split dataset
        HAR.TEST_DATA = test_image_paths
        HAR.TEST_TARGETS = test_labels

        HAR.TRAIN_DATA = train_image_paths
        HAR.TRAIN_TARGETS = train_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')
        img = Image.open(self.data[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx]



