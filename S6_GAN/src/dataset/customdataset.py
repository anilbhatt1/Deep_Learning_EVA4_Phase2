import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class Customdataset(Dataset):
  def __init__(self, data, transform=None):
    self.images, self.labels     = zip(*data)
    self.transform               = transform 

  def __len__(self):
    return len(self.images)  
    
  # Whenever we refer index from main ipynb file __getitem__ dunder method will get invoked. Also index parameter below will be a list when we are calling as batches.
  
  def __getitem__(self,index):
    if torch.is_tensor(index):
       index = index.tolist()
    
    image = Image.open(self.images[index])  
    if self.transform:
      image = self.transform(image)

    return image, self.labels[index]
