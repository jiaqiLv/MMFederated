import logging
import os
import numpy as np
import torch
import torchvision
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class Default_truncated(data.Dataset):

    def __init__(self, args, data_dir, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        """
            Generating this class too many times will be time-consuming.
            So it will be better calling this once and put it into ImageNet_truncated.
        """
        self.args = args
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        if args.dataset == 'mnist':
            self.loader = mnist_loader
        else:
            self.loader = default_loader
        if self.train:
            if args.dataset == 'GTSRB':
                self.data_dir = os.path.join(data_dir, 'GTSRB/Final_Training/Images/')
            else:
                self.data_dir = os.path.join(data_dir, 'train')
        else:
            if args.dataset == 'GTSRB':
                self.data_dir = os.path.join(data_dir, 'GTSRB/Final_Testing/Images/')
            else:
                self.data_dir = os.path.join(data_dir, 'val')

        self.all_data = torchvision.datasets.ImageFolder(root=self.data_dir, transform=self.transform).samples
        
        self.data, self.target = self.__build_truncated_dataset__()
        
    def __build_truncated_dataset__(self):
    
        data = [] 
        targets = []
        for path, target in self.all_data:
            data.append(path)
            targets.append(target)
        
        # data = torch.stack(data)
        data = np.array(data)
        targets = np.array(targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]
        
        return data, targets

    def get_data(self):
        return self.data, self.target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        
        # data = Image.open(img)
        data = self.loader(img)
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.data)
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        pass


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
def mnist_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
