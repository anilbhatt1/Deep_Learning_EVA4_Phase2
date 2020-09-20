from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# # class for Calculating and storing testing losses and testing accuracies of model for each epoch ## 
class Test_loss:

       def final_loss(bce_loss, mu, logvar):
           """
           This function will add the reconstruction loss (BCELoss) and the KL-Divergence.
           KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
           :param bce_loss: recontruction loss
           :param mu: the mean from the latent vector
           :param logvar: log variance from the latent vector
           """
           BCE = bce_loss 
           KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
           return BCE + KLD

       def test_loss_calc(self, model, device, test_loader, optimizer, criterion, epoch, max_epoch):
           self.model        = model
           self.device       = device
           self.test_loader  = test_loader
           self.optimizer    = optimizer
           self.criterion    = criterion
           self.epoch        = epoch
           self.max_epoch    = max_epoch
       
           model.eval()           
           total  = 0
                       
           with torch.no_grad():                     
                for images, labels in test_loader:
 
                    images            = images.to(device)                                    
                    reconstructed_img, mu, logvar  = model(images)                                              
                    bce_loss          = criterion(reconstruct_img, images)
                    test_loss        += final_loss(bce_loss, mu, logvar).item()                   
                    total            += images.size(0) 
                    
                test_loss   /= total  # Calculating overall test loss for the epoch                               
                print(f'Epoch: {epoch+1}/{max_epoch}, Test Loss: {test_loss:.6f}')

           return test_loss