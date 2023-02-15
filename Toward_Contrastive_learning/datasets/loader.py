import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import STL10
from torchvision import transforms
from .transform import TwoCropsTransform, GaussianBlur

class STL10DataModule(object):
    def __init__(self, data_dir="", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.train_aug1 = transforms.Compose([
                                transforms.RandomResizedCrop(96),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                transforms.ToTensor(),
                                ])
        
        self.train_aug2 = transforms.Compose([
                                transforms.RandomResizedCrop(96),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                transforms.ToTensor(),
                                ])

        self.val_aug1 = transforms.Compose([
                                transforms.ToTensor(),
                                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

        self.batch_size = batch_size
        
    def setup(self):
        stl10_all = STL10(self.data_dir, split='train+unlabeled', transform=TwoCropsTransform(self.train_aug1, self.train_aug2))
        stl10_unlabeled = STL10(self.data_dir, split='unlabeled', transform=TwoCropsTransform(self.train_aug1, self.train_aug2))
        stl10_val = STL10(self.data_dir, split='test', transform=TwoCropsTransform(self.train_aug1, self.train_aug2))
        stl10_train = STL10(self.data_dir, split='train', transform=TwoCropsTransform(self.train_aug1, self.train_aug2))
        stl10_test = STL10(self.data_dir, split='test', transform=TwoCropsTransform(self.train_aug1, self.train_aug2))
        #stl10_train = STL10(self.data_dir, split='train', transform=self.train_aug2)
        #stl10_test = STL10(self.data_dir, split='test', transform=self.val_aug1)
        return stl10_all, stl10_unlabeled, stl10_val, stl10_train, stl10_test