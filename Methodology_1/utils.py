import torch
from constants import *
import os
import torchaudio
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import uuid
import matplotlib.pyplot as plt
from lstmGan import *
import pandas as pd
from transformers import BertTokenizer, BertModel
from DLG import DLG
from encodec import EncodecModel
from torch.multiprocessing import Pool, Process, set_start_method
from GAN_utils import *
from tqdm import tqdm



def get_device():
  """ Retrieve device based on settings and availability. """
  return torch.device("cuda:0" if torch.cuda.is_available() and TRAIN_ON_GPU else "cpu")

def initialize_Bert():
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')
  return tokenizer,model


def initialize_EnCodec():
  # Instantiate a pretrained EnCodec model
  model = EncodecModel.encodec_model_48khz()
  # The number of codebooks used will be determined bythe bandwidth selected.
  # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
  # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
  # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
  # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
  model.set_target_bandwidth(6.0)
  return model


def initialize_models(device = get_device()):
  """ Initialize Generator and Discriminator models """
  generator = Generator()
  discriminator = Discriminator()

  if PRETRAINED:
    print("Loading pretraining models")
    print(GEN_PATH[-9:])
    generator.load_state_dict(torch.load(GEN_PATH))
    discriminator.load_state_dict(torch.load(DIS_PATH))



  tokenizer, bert_model = initialize_Bert()
  enCodecModel = initialize_EnCodec()
  
  # Move models to specific device
  generator.to(device)
  discriminator.to(device)

  enCodecModel.to(device)
  bert_model.to(device)

  # Return models
  return generator, discriminator, tokenizer, bert_model, enCodecModel


    
    
def make_directory_for_run():
  """ Make a directory for this training run. """
  print(f'Preparing training run {UNIQUE_RUN_ID}')
  if not os.path.exists('./runs'):
    os.mkdir('./runs')
  os.mkdir(f'./runs/{UNIQUE_RUN_ID}')


def decode_music(reconst_music, enCodecModel):
  printf("im in decode music")
  with torch.no_grad():
    wav = enCodecModel.decode(reconst_music)

  wav = wav.squeeze(0)

  return wav


def save_music(path, wav):
  printf("Saveing:",path)
  torchaudio.save(path , wav.cpu(), SAMPLE_RATE)


def generate_music(generator, enCodecModel, testloader, epoch = 0, batch = 0, device=get_device()):
  """ Generate subplots with generated examples. """
  printf("Generating")
  test_genre = None
  for batch_no, genres in enumerate(testloader, 0):
      test_genre = genres  # get a random bert sample (genre)
  
  generator.eval()
  codes, scales = generator(test_genre)


  if not os.path.exists(f'./runs/{UNIQUE_RUN_ID}/music'):
    os.mkdir(f'./runs/{UNIQUE_RUN_ID}/music')
  
  for i in range(len(test_genre)):
    reconst_music = reverse_code(codes[i], scales[i])
    wav = decode_music(reconst_music, enCodecModel)
    

    if not os.path.exists(f'./runs/{UNIQUE_RUN_ID}/music/ep{epoch}'):
      os.mkdir(f'./runs/{UNIQUE_RUN_ID}/music/ep{epoch}')

    if not os.path.exists(f'./runs/{UNIQUE_RUN_ID}/music/ep{epoch}/b{batch}'):
      os.mkdir(f'./runs/{UNIQUE_RUN_ID}/music/ep{epoch}/b{batch}')
    fname = f'./runs/{UNIQUE_RUN_ID}/music/ep{epoch}/b{batch}/epoch{epoch}_batch{batch}_sample{i}.mp3'

    save_music(fname, wav)

  printf("Generating Done")
  generator.train()

  
  
def print_training_progress(batch, generator_loss, discriminator_loss):
  """ Print training progress. """
  print('Losses after mini-batch %5d: generator %e, discriminator %e' %
        (batch, generator_loss, discriminator_loss))


def prepare_dataset(tokenizer, bert_model, enCodecModel):
  """ Prepare dataset through DataLoader """
  
  # TRAIN DATASET
  dataset = DLG(tokenizer, bert_model, enCodecModel, mode="train")
  # Batch and shuffle data with DataLoader
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
  # Return dataset through DataLoader


  # TEST DATASET
  testSet = DLG(tokenizer, bert_model, enCodecModel, mode="test")
  # Batch and shuffle data with DataLoader
  testloader = torch.utils.data.DataLoader(testSet, batch_size=TEST_SET_SIZE, shuffle=False, num_workers=0, pin_memory=False)

  return trainloader, testloader





def initialize_loss():
  """ Initialize loss function. """
  return nn.BCELoss()

def initialize_optimizers(generator, discriminator):
  """ Initialize optimizers for Generator and Discriminator. """
  generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=OPTIMIZER_LR,betas=OPTIMIZER_BETAS)
  discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=OPTIMIZER_LR,betas=OPTIMIZER_BETAS)
  return generator_optimizer, discriminator_optimizer
  

def generate_noise(batch_size = BATCH_SIZE, noise_dimension = NOISE_DIMENSION, device=None):
  """ Generate noise for number_of_images images, with a specific noise_dimension """
  code_noise =  torch.randn(batch_size, 16968, device=device)
  scale_noise =  torch.randn(batch_size, 29, device=device)
  min_val = torch.min(code_noise)
  max_val = torch.max(code_noise)

  # Scale the values down to the range of 0 to 1023
  scaled_code_noise = (code_noise - min_val) * (1023 / (max_val - min_val))

  # Round the values to the nearest integer
  int_code_noise = torch.round(scaled_code_noise).long()
  return int_code_noise,scale_noise


def efficient_zero_grad(model):
  """ 
    Apply zero_grad more efficiently
    Source: https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
  """
  for param in model.parameters():
    param.grad = None


def forward_and_backward(model, data, loss_function, targets):
  """
    Perform forward and backward pass in a generic way. Returns loss value.
  """
  outputs = model(data)
  error = loss_function(outputs, targets)
  error.backward()
  return error.item()


def perform_train_step(generator, discriminator, real_data, \
  loss_function, generator_optimizer, discriminator_optimizer, device = get_device()):
  """ Perform a single training step. """
  
  # 1. PREPARATION
  # Set real and fake labels.
  
  # Get images on CPU or GPU as configured and available
  # Also set 'actual baWWtch size', whih can be smaller than BATCH_SIZE
  # in some cases.
  real_label, fake_label = 1.0, 0.0

  codes = real_data[0].to(device)
  scales = real_data[1].to(device)
  genres = real_data[2].to(device)  

  actual_batch_size = codes.size(0)
  
  label = torch.full((actual_batch_size,1), real_label, device=device)
  
  # 2. TRAINING THE DISCRIMINATOR
  # Zero the gradients for discriminator
  efficient_zero_grad(discriminator)
  # Forward + backward on real images, reshaped
  codes = codes.view(codes.size(0), -1)
  

  error_codes = forward_and_backward(discriminator, (codes,scales), \
    loss_function, label)
  # Forward + backward on generated images
  noise_code, noise_scale = generate_noise(actual_batch_size, device=device)
  generated_codes , generated_scales = generator(genres)
  label.fill_(fake_label)
  error_generated_music =forward_and_backward(discriminator, \
    (generated_codes.detach(), generated_scales.detach()), loss_function, label)
  # Optim for discriminator
  discriminator_optimizer.step()
  
  # 3. TRAINING THE GENERATOR
  # Forward + backward + optim for generator, including zero grad
  efficient_zero_grad(generator)
  label.fill_(real_label)
  error_generator = forward_and_backward(discriminator, (generated_codes,generated_scales), loss_function, label)
  generator_optimizer.step()
  
  # 4. COMPUTING RESULTS
  # Compute loss values in floats for discriminator, which is joint loss.
  error_discriminator = error_codes + error_generated_music
  # Return generator and discriminator loss so that it can be printed.
  return error_generator, error_discriminator
  

def perform_epoch(dataloader, testloader, generator, discriminator, enCodecModel, loss_function, \
    generator_optimizer, discriminator_optimizer, epoch):
  """ Perform a single epoch. """
  for batch_no, real_data in tqdm(enumerate(dataloader, 0)):
    # Perform training step
    printf(f'Starting Batch {batch_no} epoch {epoch}...')

    generator_loss_val, discriminator_loss_val = perform_train_step(generator, \
      discriminator, real_data, loss_function, \
      generator_optimizer, discriminator_optimizer)
    # Print statistics and generate image after every n-th batch
    if batch_no % PRINT_STATS_AFTER_BATCH == 0:
      print_training_progress(batch_no, generator_loss_val, discriminator_loss_val)
      generate_music(generator, enCodecModel, testloader, epoch, batch_no)
      printf("Saving started")
      save_models(generator, discriminator, epoch, batch_no)
      printf("Saving done")
  # Save models on epoch completion.
#   save_models(generator, discriminator, epoch, "model")
  # Clear memory after every epoch
  torch.cuda.empty_cache()


def save_models(generator, discriminator, epoch , bn):
  """ Save models at specific point in time. """
  torch.save(generator.state_dict(), f'./runs/{UNIQUE_RUN_ID}/generator_{epoch}_{bn}.pth')
  torch.save(discriminator.state_dict(), f'./runs/{UNIQUE_RUN_ID}/discriminator_{epoch}_{bn}.pth')
  



def train_dcgan():
#   set_start_method('spawn')  # handling pyorch multiprocess cuda error
  """ Train the DCGAN. """
  # Make directory for unique run
  make_directory_for_run()
  
  # Set fixed random number seed
  torch.manual_seed(42)
  
  # Initialize models
  generator, discriminator , tokenizer, bert_model, enCodecModel  = initialize_models()

  # Get prepared dataset
  dataloader, testloader = prepare_dataset(tokenizer, bert_model, enCodecModel)
  
  #bert init
  # Initialize loss and optimizers
  loss_function = initialize_loss()
  generator_optimizer, discriminator_optimizer = initialize_optimizers(generator, discriminator)
  
  # Train the model
  for epoch in range(NUM_EPOCHS):
    printf(f'Starting epoch {epoch}...')
    perform_epoch(dataloader, testloader, generator, discriminator, enCodecModel, loss_function, \
      generator_optimizer, discriminator_optimizer, epoch)
  
  # Finished :)
  printf(f'Finished unique run {UNIQUE_RUN_ID}')











  






