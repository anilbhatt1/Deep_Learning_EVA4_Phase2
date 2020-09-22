import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, IMAGE_DIM, in_channel=3, out_channel=3, latent_size=100):
        super(VAE, self).__init__()
        assert IMAGE_DIM[0] % 2**4 == 0, 'Should be divided 16'
        self.init_dim = (IMAGE_DIM[0] // 2**4, IMAGE_DIM[1] // 2**4)  
        self.encode = nn.Sequential(
            # Conv-1 -> 128 -> 64
            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Conv-2 -> 64 -> 32
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Conv-3 -> 32 -> 16
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Conv-4 -> 16 -> 8
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Conv-5 -> 8 -> 4
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Conv-6 -> 4 -> 2
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # Adaptive Avg Pool 2x2x100 -> 1x1x100
        )        
        self.fc_encode = nn.Sequential(
            # reshape input, 128 -> 100
            nn.Linear(128, 100),
            #nn.Sigmoid(),
        )
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_size, self.init_dim[0]*self.init_dim[1]*32),
            nn.ReLU(),
        ) # 100 -> 2048
        self.decode = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 2048 -> 8x8x32    
            nn.ConvTranspose2d(32, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 8x8x32 -> 16x16x256
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 16x16x256 -> 32x32x128
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 32x32x128 -> 64x64x64                   
            nn.ConvTranspose2d(64, out_channel, 4, stride=2, padding=1, bias=False),
            # 64x64x64 -> 128x128x3
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample   

    def forward(self, x):
        # encoding
        x = self.encode(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encode(x)

        # get `mu` and `log_var`
        mu      = x
        log_var = x
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = z.view(z.size(0), -1)
        x = self.fc_decode(x)
        x = x.view(x.size(0), 32, self.init_dim[0], self.init_dim[1])
        x = self.decode(x)
        return x, mu, log_var  