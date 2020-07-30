from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# # class for Calculating and storing testing losses and testing accuracies of model for each epoch ## 
class Test_loss:

       def test_loss_calc(self,model, device, test_loader, optimizer, total_epoch, current_epoch, criterion, scheduler):
           self.model        = model
           self.device       = device
           self.test_loader  = test_loader
           self.optimizer    = optimizer
           self.total_epoch  = total_epoch
           self.current_epoch= current_epoch
           self.criterion    = criterion
           self.scheduler    = scheduler
       
           model.eval()
           
           correct        = 0 
           total          = 0              
           test_loss      = 0
           test_accuracy  = 0 
           test_losses    = []
           test_acc       = []
           predicted_class= []
           actual_class   = []
           wrong_predict  = []
           count_wrong    = 0 
           
           label_dict     = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
           label_total    = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
           label_correct  = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}   
                       
           with torch.no_grad():                     # For test data, we won't do backprop, hence no need to capture gradients
                for images,labels in test_loader:    # We are working in GPU, so 1 iteration will process 128 images(batch_size) in a go. Total 10,000/128 = 79 iterations will happen
 
                    images,labels    = images.to(device),labels.to(device)                                      # Images -> Tensor with shape torch.Size([128, 3, 32, 32])
                    labels_pred      = model(images)                                                            # labels_pred -> Tensor with shape torch.Size([128, 10]) 
                    #test_loss       += criterion(labels_pred, labels, reduction = 'sum').item()                # 'reduction = sum' works only for F.NLL_LOSS not for Cross_Entropy. Hence commenting out            
                    test_loss        += criterion(labels_pred, labels).item()                                   # Use torch.Tensor.item() to get a Python number from a tensor containing a single value                
                    labels_pred_max  = labels_pred.argmax(dim =1, keepdim = True)                               # labels_pred_max -> Tensor with shape torch.Size([128, 1]). We are taking maximum value out of 10 from 'labels_pred' tensor
                    correct          += labels_pred_max.eq(labels.view_as(labels_pred_max)).sum().item()        # labels -> Tensor with shape torch.Size([128]). We are changing shape of labels to ([128, 1]) for comparison purpose
                    total            += labels.size(0)                                                          # Taking number of labels in each batch size and accumulating it to get total images at end. Here labels.size(0)  = 128
                     
                    ''' labels_pred_max will look like below: torch.Size([128, 1])
                     ([[3],
                       [0],
                       [5],
                       .
                       .
                       [7]], device='cuda:0') -> 128th element
                       
                       labels will look like below: torch.Size([128])
                       ([3, 2, 5, 5, 0, 9,.....4, 4], device='cuda:0') 
                       
                       labels_pred will look like below: torch.Size([128, 10])
                       tensor([[-1.3098e+00, -5.1958e+00, -4.3112e+00,  ..., -6.5936e+00,
                                -4.1666e-01, -4.0672e+00], -> 10 elements in each row
                               [-7.6204e+00, -9.2902e+00, -4.8976e+00,  ..., -1.4079e-01,
                                -9.6599e+00, -8.5457e+00],
                                .
                                .
                               [-2.2386e+00, -3.1282e+00, -4.0142e+00,  ..., -2.4335e+00,
                                -4.5057e+00, -1.2379e+00]], device='cuda:0') -> 128th row
                      
                      * labels_pred_max.item() -> This will fail because torch.Tensor.item() is to get a Python number from a tensor containing a single value   
                      * labels.item() ->  This will fail because torch.Tensor.item() is to get a Python number from a tensor containing a single value
                      * labels_pred.item() ->  This will fail because torch.Tensor.item() is to get a Python number from a tensor containing a single value
                      * labels.view_as(labels_pred_max).item() -> This will fail because torch.Tensor.item() is to get a Python number from a tensor containing a single value
                      * if labels_pred_max == labels:  -> This will fail beacuse we are comparing different shapes                                        
                      * len(labels_pred_max) = 128 which is same as batch_size
                      * if labels_pred_max[i] == labels[i]: -> This will work because we are gathering specific elements and comparing
                      * if labels_pred_max[2] == labels[2]: -> This will work because we are gathering specific elements and comparing
                      * labels_pred_max[i] -> Will look like tensor([5], device='cuda:0') 
                      * labels[i] -> Will look like tensor(2, device='cuda:0')
                      * labels[i].item() -> Will work & return an integer
                      * labels_pred_max[i].item() -> Will work & return an integer
                    '''
                    
                    #if current_epoch == (total_epoch - 1): 
                    #       for i in range(len(labels_pred_max)):
                    #           counter_key = ' '
                    #           counter_key = label_dict.get(labels[i].item())   # Getting labels from 'label_dict'
                    #           label_total[counter_key] += 1                    # Increasing total count of corresponding label

                    #          if labels_pred_max[i] == labels[i]:
                    #              label_correct[counter_key] += 1               # Increasing correct count of corresponding label
                    #           else:    
                    #              if count_wrong   < 26:                                            # Capturing 26 wrongly predicted images for last epoch
                    #                 wrong_predict.append(images[i])                                # with its predicted and actual class 
                    #                 predicted_class.append(labels_pred_max[i].item())
                    #                 actual_class.append(labels[i].item())
                    #                 count_wrong += 1
              
                test_loss   /= total  # Calculating overall test loss for the epoch
                test_losses.append(test_loss)    
                                  
                test_accuracy =  (correct/total)* 100
                test_acc.append(test_accuracy)      
                
                #'scheduler.step' is required to update LR while using a scheduler like ReduceLROnPlateau. Here metric used is 'test_loss'
                scheduler.step(test_loss)   
               
                lr = 0 
                if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                   lr = self.scheduler.get_last_lr()[0]           # Won't work for ReduceLROnPlateau
                else:
                   lr = self.optimizer.param_groups[0]['lr']     
                             
                print('\nTest set: Average loss: {:.4f}, Test Accuracy: {:.2f}, LR : {:.6f}' .format(test_loss, test_accuracy, lr))

           return test_losses, test_acc, wrong_predict, predicted_class, actual_class, label_total, label_correct
