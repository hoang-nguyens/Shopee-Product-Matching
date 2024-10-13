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

from image_embedding import EmbeddingNet

class ShopeeDataset(Dataset):
    def __init__(self, df, img_dirs, transforms = None):
        super(ShopeeDataset, self).__init__()
        self.df = df
        self.img_dirs = img_dirs
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ids):
        image_names = self.df.iloc[ids, 1]
        image_titles = self.df.iloc[ids, 3]

        image_paths = self.img_dirs + image_names
        images = Image.open(image_paths)
        if self.transforms is not None:
            images = self.transforms(images)

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

def getDataloader(df, img_dir, batch_size, shuffle = True, transforms = None):
    dataset = ShopeeDataset(df, img_dir, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=shuffle, num_workers=4)
    return dataloader



if __name__ == '__main__':
    pass
    df = pd.read_csv('train.csv')
    img_dirs = 'train_images/'
    dataset = ShopeeDataset(df = df, img_dirs = img_dirs, transforms = None)
    dataloader = getDataloader(df, img_dirs, 16, False, transforms=getTransform())

    model_name = 'nfnet_f0'
    model = EmbeddingNet(model_name)
    #print(model)
    i = 0
    for image, title in dataloader:
        out = model(image)
        print(out[0])
        i += 1
        if(i >= 3) : break







