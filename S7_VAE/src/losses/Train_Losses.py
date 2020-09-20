from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
from time import time
from datetime import datetime 

# # class for Calculating and storing training losses and training accuracies of model for each batch per epoch ## 
class Train_loss:

      def get_sample_image(recontructed_img):
          """
          modify sample images for saving purpose
          """
          y_hat = recontructed_img.view(32, 3, 128, 128).permute(0, 2, 3, 1)    #Modify axes to (32, 28, 28, 3) via permute. 32 -> batch_size
          result = (y_hat.detach().cpu().numpy()+1)/2.
          return result

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
        
      def train_loss_calc(self,model, device, train_loader, optimizer, epoch, max_epoch, criterion):
            
          self.model        = model
          self.device       = device
          self.train_loader = train_loader
          self.optimizer    = optimizer   
          self.epoch        = epoch
          self.max_epoch    = max_epoch
          self.criterion    = criterion
          
          model.train()
          pbar = tqdm(train_loader)  # Wrapping train_loader in tqdm to show progress bar for each epoch while training         
          img_list = []
        
          for batch_idx, data in enumerate(pbar,0):
                        
              images, labels = data        
                       
              images      = images.to(device)       # Moving images and correspondig labels to GPU
              optimizer.zero_grad()                 # Zeroing out gradients at start of each batch so that backpropagation won't take accumulated value
              reconstructed_img, mu, logvar = model(images)  # Calling CNN model to predict the images
              bce_loss    = criterion(reconstructed_img, images)
              train_loss  = final_loss(bce_loss, mu, logvar)            
            
              # Backpropagation
              train_loss.backward()
              optimizer.step()       
              pbar.set_description(desc=f'Train Loss = {train_loss.item():0.6f} Epoch = {epoch} Batch Id = {batch_idx}')

          if epoch % 10 == 0 or ((epoch == max_epoch-1) and (batch_idx == len(train_loader)-1)):  ### Saving reconstructed image to drive
            print(f'Epoch: {epoch+1}/{max_epoch}, Train Loss: {train_loss.item():.6f}')
            img = get_sample_image(recontructed_img)
            t = datetime.now()
            time_stamp = t.strftime("%Y")+t.strftime("%m")+t.strftime("%d")+t.strftime("%H")+t.strftime("%M")+t.strftime("%S")
            imsave(f'/content/gdrive/My Drive/EVA4P2_S7_Training/Images_Size128_D0920/VAE_{epoch+1}_{time_stamp}.jpg', img[0])          
              

          if epoch % 45 == 0 or (epoch == (max_epoch-1) and (batch_idx == len(train_loader)-1)): ### Keep the model in Gpu & Save the model values in intermittent epochs
              t = datetime.now()
              time_stamp = t.strftime("%Y")+t.strftime("%m")+t.strftime("%d")+t.strftime("%H")+t.strftime("%M")+t.strftime("%S")         
              torch.save(model.state_dict(),f'/content/gdrive/My Drive/EVA4P2_S7_Training/Model_Weights_D0920/VAE_GPU_{epoch}_{time_stamp}.pt')
              print(f'GPU model saved in epoch {epoch+1}/{max_epoch}')
              
              save_img = recontructed_img.detach().cpu()
              img_list.append(vutils.make_grid(save_img, padding=2, normalize=True))  ### img_list will be used to create animation at the end of training

          if (epoch == (max_epoch-1) and (batch_idx == len(train_loader)-1)):    ### Convert the model to CPU & save the model values on final epoch              
              t = datetime.now()
              time_stamp = t.strftime("%Y")+t.strftime("%m")+t.strftime("%d")+t.strftime("%H")+t.strftime("%M")+t.strftime("%S")
              model.eval()
              model.to('cpu')
              traced_model = torch.jit.trace(model,torch.randn(1,3,128,128))      
              traced_model.save(f'/content/gdrive/My Drive/EVA4P2_S7_Training/Model_Weights_D0920/VAE_CPU_{epoch}_{time_stamp}.pt')
              print(f' **** CPU model Saved in epoch:{epoch+1}/{max_epoch}')
              model.to(device)      
              model.train()          
              
          return train_loss.item() , img_list  