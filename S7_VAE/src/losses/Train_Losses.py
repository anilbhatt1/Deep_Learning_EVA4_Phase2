from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, imsave
import cv2
import torchvision
import torchvision.utils as vutils 

# # class for Calculating and storing training losses and training accuracies of model for each batch per epoch ## 
class Train_loss:

      def draw_and_save(self, tensors, name, figsize=(15,15), *args, **kwargs):
            
          grid_tensor = torchvision.utils.make_grid(tensors, *args, **kwargs)
          grid_image  = grid_tensor.permute(1, 2, 0)
          plt.figure(figsize = figsize)
          plt.imshow(grid_image)
          plt.xticks([])
          plt.yticks([])

          plt.savefig(name, bbox_inches='tight')
          plt.close()
          flag = True
          return flag   

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
          KLD = KLD /(batch_sz * channels * img_dim1 * img_dim2)
          return MSE + KLD    

      def get_sample_image(self, reconstructed_img):
          """
          modify sample images for saving purpose
          """
          cnt = reconstructed_img.shape
          sample_img = reconstructed_img.view(cnt, 3, 128, 128).permute(0, 2, 3, 1)    #Modify axes to (32, 28, 28, 3) via permute. 32 -> batch_size
          result = (sample_img.detach().cpu().numpy()+1)/2.
          return result
      
      def train_loss_calc(self,model, device, train_loader, optimizer, epoch, max_epoch, criterion):
            
          self.model        = model
          self.device       = device
          self.train_loader = train_loader
          self.optimizer    = optimizer   
          self.epoch        = epoch
          self.max_epoch    = max_epoch
          self.criterion    = criterion
          
          model.train()      
          img_list = []
          flg = 0
          path_name    = '/content/gdrive/My Drive/EVA4P2_S7_Training/Images_Size128_D0922/'
          path_name_wt = '/content/gdrive/My Drive/EVA4P2_S7_Training/Model_Weights_D0922/'
          
          for batch_idx, data in enumerate(train_loader):
               
              images =  data[0].to(device)      # Moving images and correspondig labels to GPU
              optimizer.zero_grad()                 # Zeroing out gradients at start of each batch so that backpropagation won't take accumulated value
              reconstructed_img, mu, logvar = model(images)  # Calling CNN model to predict the images
              mse_loss    = criterion(reconstructed_img, images)
              batch_sz, channels, img_dim1, img_dim2 = images.shape[0], images.shape[1], images.shape[2], images.shape[3]
              train_loss1  = self.final_loss(mse_loss, mu, logvar, batch_sz, channels, img_dim1, img_dim2)          

              # Backpropagation
              train_loss1.backward()
              optimizer.step()       
              
              ### Saving reconstructed image to drive. Batch_idx selected is pen-ultimate in an epoch to ensure entire batch_size is captured
              if (epoch % 30 == 0 and (batch_idx == len(train_loader)-2)) or (epoch == (max_epoch-1) and (batch_idx == len(train_loader)-2)):                                   
                  img = self.get_sample_image(reconstructed_img)
                  t = datetime.now()
                  time_stamp = t.strftime("%Y")+t.strftime("%m")+t.strftime("%d")+t.strftime("%H")+t.strftime("%M")+t.strftime("%S")
                  num = random.randint(1,30)
                  imsave(f'{path_name}VAE_{epoch+1}_{time_stamp}.jpg', img[num])
                  flg = self.draw_and_save(images.detach().cpu(), f'{path_name}VAE_{epoch}_input_{time_stamp}.jpg')
                  flg = self.draw_and_save(reconstructed_img.detach().cpu(), f'{path_name}VAE_{epoch}_output_{time_stamp}.jpg')
                  print(f' Sample image-{num} Saved - Epoch: {epoch+1}/{max_epoch}, Train Loss: {train_loss1.item():.6f}, batch_idx:{batch_idx}')          
              
              ### Keep the model in Gpu & Save the model values in intermittent epochs
              if (epoch % 45 == 0 and (batch_idx == len(train_loader)-2)) or (epoch == (max_epoch-1) and (batch_idx == len(train_loader)-2)): 
                  t = datetime.now()
                  time_stamp = t.strftime("%Y")+t.strftime("%m")+t.strftime("%d")+t.strftime("%H")+t.strftime("%M")+t.strftime("%S")         
                  torch.save(model.state_dict(),f'{path_name_wt}VAE_GPU_{epoch}_{time_stamp}.pt')                 
              
                  save_img = reconstructed_img.detach().cpu()
                  img_list.append(vutils.make_grid(save_img, padding=2, normalize=True))  ### img_list will be used to create animation at the end of training
                  print(f'GPU model saved in epoch {epoch+1}/{max_epoch}, batch_idx:{batch_idx}')

              if (epoch == (max_epoch-1) and (batch_idx == len(train_loader)-1)):    ### Convert the model to CPU & save the model values on final epoch              
                  t = datetime.now()
                  time_stamp = t.strftime("%Y")+t.strftime("%m")+t.strftime("%d")+t.strftime("%H")+t.strftime("%M")+t.strftime("%S")
                  model.eval()
                  model.to('cpu')
                  traced_model = torch.jit.trace(model,torch.randn(1,3,128,128))      
                  traced_model.save(f'{path_name_wt}VAE_CPU_{epoch}_{time_stamp}.pt')
                  print(f' **** CPU model Saved in epoch:{epoch+1}/{max_epoch}, batch_idx:{batch_idx}')
                  model.to(device)      
                  model.train()          
              
          return train_loss1.item() , img_list  