from torch import nn
from constants import *
import sys



class Discriminator(nn.Module):
  def __init__(self, seq_shape = SEQ_SHAPE):
    super(Discriminator, self).__init__()

    self.seq_shape = seq_shape

    # self.lstm1 = nn.LSTM(input_size=seq_shape[1], hidden_size=512, bidirectional=False, batch_first=True)
    self.lstm1 = nn.LSTM(input_size=seq_shape, hidden_size=512, bidirectional=False, batch_first=True)
    self.lstm2 = nn.LSTM(input_size=512, hidden_size=512, bidirectional=True, batch_first=True)
    self.dense1 = nn.Linear(1024, 512)
    self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2)
    self.dense2 = nn.Linear(512, 512)
    self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2)
    self.dense3 = nn.Linear(512, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, seq):
    printd("disc input", seq.shape)
    lstm1_out, _ = self.lstm1(seq)
    printd("LSTM1", lstm1_out.shape)
    lstm2_out, _ = self.lstm2(lstm1_out)
    printd("LSTM2", lstm2_out.shape)
    dense1_out = self.dense1(lstm2_out)
    leakyrelu1_out = self.leakyrelu1(dense1_out)
    dense2_out = self.dense2(leakyrelu1_out)
    leakyrelu2_out = self.leakyrelu2(dense2_out)
    dense3_out = self.dense3(leakyrelu2_out)
    validity = self.sigmoid(dense3_out)

    return validity


class Generator(nn.Module):
  def __init__(self, latent_dim = LATENT_DIM, seq_shape = SEQ_SHAPE):
    super(Generator, self).__init__()

    self.latent_dim = latent_dim
    self.seq_shape = seq_shape

    self.lstm1 = nn.LSTM(input_size=self.latent_dim, hidden_size=512, bidirectional=False, batch_first=True)
    self.lstm2 = nn.LSTM(input_size=512, hidden_size=512, bidirectional=True, batch_first=True)
    self.dense1 = nn.Linear(1024, 256)
    self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2)
    self.batchnorm1 = nn.BatchNorm1d(256, momentum=0.8)
    self.dense2 = nn.Linear(256, 512)
    self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2)
    self.batchnorm2 = nn.BatchNorm1d(512, momentum=0.8)
    self.dense3 = nn.Linear(512, 1024)
    self.leakyrelu3 = nn.LeakyReLU(negative_slope=0.2)
    self.batchnorm3 = nn.BatchNorm1d(1024, momentum=0.8)
    # self.dense4 = nn.Linear(1024, seq_shape[0] * seq_shape[1])
    self.dense4 = nn.Linear(1024, seq_shape)
    self.tanh = nn.Tanh()

  def forward(self, noise):
    printd("Gen input", noise.shape)
    
    lstm1_out, _ = self.lstm1(noise)
    printd("LSTM1", lstm1_out.shape)
    
    lstm2_out, _ = self.lstm2(lstm1_out)
    printd("LSTM2", lstm2_out.shape)
    
    dense1_out = self.dense1(lstm2_out)
    leakyrelu1_out = self.leakyrelu1(dense1_out)
    batchnorm1_out = self.batchnorm1(leakyrelu1_out)
    printd("dense1_out", batchnorm1_out.shape)
    
    dense2_out = self.dense2(batchnorm1_out)
    leakyrelu2_out = self.leakyrelu2(dense2_out)
    batchnorm2_out = self.batchnorm2(leakyrelu2_out)
    printd("dense2_out", batchnorm2_out.shape)
    
    dense3_out = self.dense3(batchnorm2_out)
    leakyrelu3_out = self.leakyrelu3(dense3_out)
    batchnorm3_out = self.batchnorm3(leakyrelu3_out)
    printd("dense3_out", batchnorm3_out.shape)
    
    dense4_out = self.dense4(batchnorm3_out)
    tanh_out = self.tanh(dense4_out)
    printd("dense4_out", tanh_out.shape)
    
    seq = tanh_out.view(-1, self.seq_shape)
    printd("final sequence", seq.shape)

    return seq