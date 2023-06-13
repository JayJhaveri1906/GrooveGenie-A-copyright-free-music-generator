import torch
from torch import nn
from constants import *
from GAN_utils import reverse_code

class Generator(nn.Module):
  """
    Vanilla GAN Generator
  """
  def __init__(self):
    super(Generator, self).__init__()
    self.layers = nn.Sequential(
      nn.Linear(768, 128, bias=False),
      nn.BatchNorm1d(128, 0.8),
      nn.LeakyReLU(),

      nn.Linear(128, 256, bias=False),
      nn.BatchNorm1d(256, 0.8),
      nn.LeakyReLU(),

      nn.Linear(256, 512, bias=False),
      nn.BatchNorm1d(512, 0.8),
      nn.LeakyReLU(),

    )

    self.codes = nn.Sequential(
      nn.Linear(512, 16968, bias=False),
      nn.BatchNorm1d(16968, 0.8),
      nn.ReLU()
    )
  
    self.scale_node = nn.Sequential(
      nn.Linear(512, 29, bias=False),
      nn.BatchNorm1d(29, 0.8),
      nn.ReLU()
    )

  def forward(self, x):
    """Forward pass"""
    # print(x.shape)
    x = self.layers(x)
    codes = self.codes(x)
    # codes = (codes * 10**5).long()
    min_val = torch.min(codes)
    max_val = torch.max(codes)

    # Scale the values down to the range of 0 to 1023
    scaled_codes = (codes - min_val) * (1023 / (max_val - min_val))

    # Round the values to the nearest integer
    int_codes = torch.round(scaled_codes).long()
    scales = self.scale_node(x)

    # reconstructed = reverse_code(codes,scales)
    return int_codes,scales


class Discriminator(nn.Module):
  """
    Vanilla GAN Discriminator
  """
  def __init__(self):
    super(Discriminator, self).__init__()
    self.layers_codes = nn.Sequential(
      nn.Linear(16968, 4096), 
      nn.LeakyReLU(),
      nn.Linear(4096, 1024), 
      nn.LeakyReLU(),
      nn.Linear(1024, 512), 
      nn.LeakyReLU(),
      nn.Linear(512, 256), 
      nn.LeakyReLU(),
      nn.Linear(256, 128), 
      nn.LeakyReLU(),
      
    )
    self.layers_scales = nn.Sequential(
      nn.Linear(29, 64), 
      nn.LeakyReLU(),
      nn.Linear(64, 128), 
      nn.LeakyReLU(),    
    )

    self.decision = nn.Sequential(
      nn.Linear(128,1),
      nn.Sigmoid(),
    )

  def forward(self, x):
    """Forward pass"""
    x_codes = self.layers_codes(x[0].float()) # CHECK DTYPE HERE
    x_scales = self.layers_scales(x[1].float())
    x = x_codes * x_scales
    x = self.decision(x)
    return x

