from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable

def gaussian(window_size, sigma):    # Creates a gaussian tensor of size 11 eg: torch.Size([11])
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # Adds one more dimension to gaussian tensor - torch.Size([11, 1])
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)   # mm is matrix multiplication. Also adds 2 more dimensions - torch.Size([1, 1, 11, 11])

    # expand     -> Returns a new view of the self tensor with singleton dimensions expanded to a larger size
    # contiguous -> It is like transpose but with seperate memory
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous()) # Changes window size as torch.Size([3, 1, 11, 11])                                                                                     
    return window

# Example used to explain comments below: img1.size (sr or Fake) -> torch.Size([1, 3, 144, 144]), img2.size (hr or GT) -> torch.Size([1, 3, 144, 144])

def _ssim(img1, img2, window, window_size, channel, size_average=True):    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)   # Conv2d parms: torch.Size([1, 3, 144, 144]), torch.Size([3, 1, 11, 11]), padding= 5, groups = 3
                                                                             # conv2d returns mu1 & mu2 of size torch.Size([1, 3, 144, 144])    
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)       # mu1_sq and mu2_sq sizes : torch.Size([1, 3, 144, 144])
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq  # sigma1_sq.size: torch.Size([1, 3, 144, 144])
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq  # sigma2_sq.size: torch.Size([1, 3, 144, 144])
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))  # ssim_map.size : torch.Size([1, 3, 144, 144])

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size=11, size_average=True):  # This is the function that is called while training which calls -> create_window -> _ssim (img1 is sr, img2 is hr/GT)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)   # type_as -> Returns this tensor cast to the type of the given tensor.

    return _ssim(img1, img2, window, window_size, channel, size_average)