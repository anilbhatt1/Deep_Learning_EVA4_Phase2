import torch
from torch import nn
from torchvision.models.vgg import vgg16

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss -> Calculates difference between 1 and value returned by discriminator (fake_out) after evaluating the fake img generated by generator
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss -> This is VGG loss between fake_img (sr) and real_img(hr)
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss -> This is MSE loss between fake img and real_img
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss  -> Total Variation Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss

# TVLoss -> Total Variation Loss
# The total variation is the sum of the absolute differences for neighboring pixel-values in the input images. This measures how much noise is in the images.
# TV loss is getting fake_img generated(sr_img) as input

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])                          # If sr_img size is (64, 3, 44, 44), we are passing tensor as (64,3,43,44) i.e discarding first row. This is to facilitate vertical grad calc.
        count_w = self.tensor_size(x[:, :, :, 1:])                          # If sr_img size is (64, 3, 44, 44), we are passing tensor as (64,3,44,43) i.e discarding first column. This is to facilitate horizontal grad calc.
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()  # Difference of neighbouring pixel values using rows i.e. Calculating vertical gradient.(Refer EVA4-P1-S1)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()  # Difference of neighbouring pixel values using columns i.e. calculation horizontal gradient.
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):     
        return t.size()[1] * t.size()[2] * t.size()[3]        