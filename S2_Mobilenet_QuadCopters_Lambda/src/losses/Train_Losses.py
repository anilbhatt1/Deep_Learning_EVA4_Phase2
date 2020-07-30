from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# # class for Calculating and storing training losses and training accuracies of model for each batch per epoch ## 
class Train_loss:
        
      def train_loss_calc(self,model, device, train_loader, optimizer, epoch, factor, criterion, scheduler =None, print_idx=0, maxlr=0):
            
          self.model        = model
          self.device       = device
          self.train_loader = train_loader
          self.optimizer    = optimizer   
          self.epoch        = epoch
          self.factor       = factor
          self.criterion    = criterion      
          self.scheduler    = scheduler
          self.print_idx    = print_idx
          self.maxlr        = maxlr    
          self.metric       = 0      
          
          model.train()
          pbar = tqdm(train_loader)  # Wrapping train_loader in tqdm to show progress bar for each epoch while training          
            
          correct             = 0
          total               = 0
          train_losses        = []
          train_acc           = []
          
          for batch_idx, data in enumerate(pbar,0):
                        
              images, labels = data        
                       
              images, labels = images.to(device), labels.to(device)   # Moving images and correspondig labels to GPU
              optimizer.zero_grad()  # Zeroing out gradients at start of each batch so that backpropagation won't take accumulated value
              labels_pred = model(images)  # Calling CNN model to predict the images
              loss = criterion(labels_pred, labels)   # Calculating Negative Likelihood Loss by comparing prediction vs ground truth
              
              # Applying L1 regularization to the training loss calculated
              L1_criterion = nn.L1Loss(size_average = None, reduce = None, reduction = 'mean')
              reg_loss     = 0
              for param in model.parameters():
                zero_tensor = torch.rand_like(param) * 0 # Creating a zero tensor with same size as param
                reg_loss    += L1_criterion(param, zero_tensor)
              loss += factor * reg_loss 
              
            
              # Backpropagation
              loss.backward()
              optimizer.step()
              
              # This was used for OneCycle LR policy. 'get_last_lr()' giving error while using ReduceLRonPlateau hence commenting out 
              #lr = self.scheduler.get_last_lr()[0] if self.scheduler else (self.optimizer.lr_scheduler.get_last_lr()[0] if self.optimizer.lr_scheduler else self.optimizer.param_groups[0]['lr'])
              
              lr = 0
              if scheduler and not (isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                 lr = self.scheduler.get_last_lr()[0]   # Won't work for ReduceLRonPlateau
              else:
                 lr = self.optimizer.param_groups[0]['lr']  
  
              # Calculating accuracies
              labels_pred_max = labels_pred.argmax(dim = 1, keepdim = True) # Getting the index of max log probablity predicted by model
              correct         += labels_pred_max.eq(labels.view_as(labels_pred_max)).sum().item() # Getting count of correctly predicted
              total           += len(images) # Getting count of processed images
              train_acc_batch = (correct/total)*100            
              pbar.set_description(desc=f'Train Loss = {loss.item()} Batch Id = {batch_idx} Train Acc = {train_acc_batch:0.2f} LR ={lr:0.6f}')                                       
     
          train_acc.append(train_acc_batch)  # To capture only final batch accuracy of an epoch
          train_losses.append(loss)          # To capture only final batch loss of an epoch
        
          return train_losses, train_acc         
