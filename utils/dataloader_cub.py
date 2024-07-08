import torch
import numpy as np


import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image, ImageFile
import torchvision.transforms as transforms
    
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


def query_transform():
    """
    Query images transform.

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


def load_data_CLASS(root, batch_size, num_workers):
    Cub2011_UC.init(root)
    test_dataset = Cub2011_UC(root, 'test', query_transform())
    train_dataset = Cub2011_UC(root, 'train', train_transform())
    
    print(len(test_dataset))
    print(len(train_dataset))

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size*4,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size*4,
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
    

    return train_dataloader,val_dataloader,test_dataloader




class Cub2011_UC(Dataset):
    base_folder = 'images/'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader

        if mode == 'train':
            self.data = Cub2011_UC.TRAIN_DATA
            self.targets = Cub2011_UC.TRAIN_TARGETS
        elif mode == 'test':
            self.data = Cub2011_UC.TEST_DATA
            self.targets = Cub2011_UC.TEST_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')



    @staticmethod
    def init(root):
        images = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        all_data = data.merge(train_test_split, on='img_id')
        all_data['filepath'] = 'images/' + all_data['filepath']
        train_data = all_data[all_data['is_training_img'] == 1]
        test_data = all_data[all_data['is_training_img'] == 0]


        # Split dataset
        Cub2011_UC.TEST_DATA = test_data['filepath'].to_numpy()
        Cub2011_UC.TEST_TARGETS = (test_data['target']-1).to_numpy()

        Cub2011_UC.TRAIN_DATA = train_data['filepath'].to_numpy()
        Cub2011_UC.TRAIN_TARGETS = (train_data['target']-1).to_numpy()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx]


