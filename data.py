# Dataset Class: return image, text
# Dataloader function
# transform image function (not embedding)
# image path, split data # image name in csv file
# handle csv file: create group column ( set of similar images)
# use phash as first grouping approach # in visualization file

import torch
import torchvision
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, Normalize, ToTensor, CenterCrop
from torchvision.io import read_image
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset



class ShopeeDataset(Dataset):
    def __init__(self, df, img_dirs, transforms = None, have_label = False):
        super(ShopeeDataset, self).__init__()
        self.df = df
        self.img_dirs = img_dirs
        self.transforms = transforms
        self.have_label = have_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ids):
        image_names = self.df.iloc[ids, 1]
        image_titles = self.df.iloc[ids, 3]
        if self.have_label:
            image_labels = self.df.iloc[ids, 5]

        image_paths = self.img_dirs + image_names
        images = Image.open(image_paths)
        if self.transforms is not None:
            images = self.transforms(images)
        if self.have_label == True:
            return images, image_titles, image_labels
        else:
            return images, image_titles

def getTransform():
    return Compose([
        Resize((224 + 32, 224 + 32), interpolation= Image.BICUBIC),
        CenterCrop((224, 224)),
        # RandomHorizontalFlip(0.5),
        # RandomVerticalFlip(0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def getDataloader(df, img_dir, batch_size, shuffle = True, transforms = None, have_label = False):
    dataset = ShopeeDataset(df, img_dir, transforms=transforms, have_label= have_label)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=shuffle, num_workers=4)
    return dataloader







