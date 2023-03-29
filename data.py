import PIL
import numpy as np
import sys
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import cv2
from source_target_transforms import *

class DataSampler:
    def __init__(self, img, target, crop_size):
        self.img = img
        self.target = target
        self.pairs = self.target, self.img

        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(crop_size),
            ToTensor()]) 

    def create_hr_lr_pairs(self):
        return (self.img, self.target)

    def generate_data(self):
        while True:
            gt, inp = self.pairs
            gt_tensor, inp_tensor = self.transform((gt, inp))
            gt_tensor = torch.unsqueeze(gt_tensor, 0)
            inp_tensor = torch.unsqueeze(inp_tensor, 0)
            yield gt_tensor, inp_tensor
