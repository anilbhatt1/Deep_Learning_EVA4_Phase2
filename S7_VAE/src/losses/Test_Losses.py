from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# # class for Calculating and storing testing losses and testing accuracies of model for each epoch ## 
class Test_loss:

       def final_loss(self, mse_loss, mu, logvar, batch_sz, channels, img_dim1, img_dim2):
           """
           This function will add the reconstruction loss (BCELoss) and the KL-Divergence.
           KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
           :param bce_loss: recontruction loss
           :param mu: the mean from the latent vector
           :param logvar: log variance from the latent vector
           """
           MSE = mse_loss 
           KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
           KLD = KLD/(batch_sz * channels * img_dim1 * img_dim2)
           return MSE + KLD

       def test_loss_calc(self, model, device, test_loader, optimizer, criterion, epoch, max_epoch):
           self.model        = model
           self.device       = device
           self.test_loader  = test_loader
           self.optimizer    = optimizer
           self.criterion    = criterion
           self.epoch        = epoch
           self.max_epoch    = max_epoch
       
           model.eval()           
           total      = 0
           test_loss1 = 0
                       
           with torch.no_grad():                     
                for images, labels in test_loader:
 
                    images            = images.to(device)                                    
                    reconstructed_img, mu, logvar  = model(images)                                              
                    mse_loss          = criterion(reconstructed_img, images)
                    test_loss1       += self.final_loss(mse_loss, mu, logvar, batch_sz, channels, img_dim1, img_dim2).item()                   
                    total            += images.size(0) 
                    
                test_loss1   /= total  # Calculating overall test loss for the epoch                               

           return test_loss1
