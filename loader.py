import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import os 
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image


class ImageMaskGenerator(Dataset):

    def __init__(self, path, transform = None):
        self.path = path
        self.transform = transform
        self.image_names = [i for i in os.listdir(path) if not ((i.endswith('_mask.png') or i.endswith('_mask_1.png')) or i.endswith('mask_2.png'))]


    def __len__(self):

        return len(self.image_names)


    def __getitem__(self, index):

        image_name = self.image_names[index]
        mask_name = image_name.replace('.png', '_mask.png')

        img_path = os.path.join(self.path, image_name)
        mask_path  = os.path.join(self.path, mask_name)

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform:

            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask