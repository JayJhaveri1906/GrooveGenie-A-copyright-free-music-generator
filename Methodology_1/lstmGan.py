import torch
from torch import nn
from constants import *
from GAN_utils import reverse_code

class Generator(nn.Module):
  """
    Vanilla GAN Generator
  """
  def __init__(self, noise_size = 100, bert_size = 768, hidden_units = 256, output_size = 16968, drop_prob = 0.6):
    super(Generator, self).__init__()


    self.noise_size = noise_size
    self.hidden_dim = hidden_units
    self.dropout = nn.Dropout(p=drop_prob)
    self.num_layers = 2
    self.bert_size = bert_size
    self.output_size = output_size


    # Bidirectional LSTMs
    self.bi_lstm1 = nn.LSTM(
            input_size= self.noise_size+self.bert_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
    )
    self.bi_lstm2 = nn.LSTM(
            input_size=self.hidden_dim*2,
            hidden_size=self.hidden_dim*2,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
    )

    self.fc1 = nn.Linear(self.hidden_dim*4, self.hidden_dim*4)

    self.fc_codes = nn.Linear(self.hidden_dim*4, self.output_size)
    self.fc_scales = nn.Linear(self.hidden_dim*4, 29)

  def forward(self, bert_embedding):
    """Forward pass"""
    #noise
    real_batch_size = bert_embedding.size(0)
    z = torch.randn(real_batch_size, self.noise_size).to(get_device())
    # Conditional GAN on bert_embeddings of genre
    x = torch.cat([z, bert_embedding], dim=-1)


    # Initialize initial hidden states
    weight = next(self.parameters()).data
    layer_mult = 2 # for being bidirectional

    device = get_device()
    h0 = weight.new(self.num_layers * layer_mult, self.hidden_dim).zero_().to(device)
    c0 = weight.new(self.num_layers * layer_mult, self.hidden_dim).zero_().to(device)
    hidden_states = (h0, c0)

    x, states = self.bi_lstm1(x, hidden_states)

    h0 = weight.new(self.num_layers * layer_mult, self.hidden_dim*2).zero_().to(device)
    c0 = weight.new(self.num_layers * layer_mult, self.hidden_dim*2).zero_().to(device)
    hidden_states = (h0, c0)
    x, states_ = self.bi_lstm2(x, hidden_states)

    x = self.fc1(x)

    # codes
    codes = self.fc_codes(x)

    ## Scaling codes to integers and maximum val of 1023 (ENCODEC compatibility issues)
    min_val = torch.min(codes)
    max_val = torch.max(codes)

    # Scale the values down to the range of 0 to 1023
    scaled_codes = (codes - min_val) * (1023 / (max_val - min_val))

    # Round the values to the nearest integer
    int_codes = torch.round(scaled_codes).long()
    

    # scales
    scales = self.fc_scales(x)

    return int_codes, scales


class Discriminator(nn.Module):
  """
    Vanilla GAN Discriminator
  """
  def __init__(self, input_size = 16968, hidden_units = 256, drop_prob = 0.6):
    super(Discriminator, self).__init__()

    self.input_size = input_size
    self.hidden_dim = hidden_units
    self.dropout = nn.Dropout(p=drop_prob)
    self.num_layers = 2

    self.bi_lstm_codes = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=drop_prob)
    

    self.bi_lstm_scales = nn.LSTM(
                input_size=600*29,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=drop_prob)
    
    self.fc = nn.Linear(self.hidden_dim*4, 1)  # *2 concat *2 = *4  ## *2 cause bidrec
    
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    """Forward pass"""


    # Initialize initial hidden states
    weight = next(self.parameters()).data
    layer_mult = 2 # for being bidirectional

    device = get_device()
    h0 = weight.new(self.num_layers * layer_mult, self.hidden_dim).zero_().to(device)
    c0 = weight.new(self.num_layers * layer_mult, self.hidden_dim).zero_().to(device)
    hidden_states = (h0, c0)

    # Extracting codes and scales
    x_codes = x[0].float() # CHECK DTYPE HERE
    x_scales = x[1].float()
    
    # Codes
    x_codes = self.dropout(x_codes)
    x_codes, state = self.bi_lstm_codes(x_codes, hidden_states)

    # Scales
    x_scales = x_scales.unsqueeze(-1).repeat(1, 600, 1).squeeze()
    x_scales = self.dropout(x_scales)
    x_scales, _= self.bi_lstm_scales(x_scales)

    # Combine both
    out = torch.cat((x_codes, x_scales), dim=-1)

    # pass through linear output layer with sigmoid activation and return output
    out = self.sigmoid(self.fc(out))

    return out