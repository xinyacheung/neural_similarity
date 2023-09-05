import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

class LSTMDataset(Dataset):

  def __init__(self, X, Y, C ):
    self.input = X
    self.output = Y
    self.label = C

  def __getitem__(self, index):

    return self.input[index], self.output[index], self.label[index]
    
  def __len__(self):
    return self.input.size(0)

