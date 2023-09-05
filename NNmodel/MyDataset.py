import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF

from PIL import Image

class CNNDataset(Dataset):

  def __init__(self, path:list, classes:list, transform=None):

    self.images = path
    self.label = classes

  def __getitem__(self, index):

    img = TF.to_tensor(Image.open(self.images[index]).convert('RGB'))
    train_X = img
    train_Y = self.label[index]

    return train_X, train_Y

  def __len__(self):
    return len(self.images)

class LSTMDataset(Dataset):

  def __init__(self, X, Y, C ):
    self.input = X
    self.output = Y
    self.label = C

  def __getitem__(self, index):

    return self.input[index], self.output[index], self.label[index]
    
  def __len__(self):
    return self.input.size(0)

