import torch
from constants import *
import os
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from lstmGan import Discriminator, Generator
from DLG import DLG
from torch.multiprocessing import Pool, Process, set_start_method
from tqdm import tqdm
from midiHandler import MIDI
from genMusic import generate_music, generate_noise
import wandb



def initialize_models(device = get_device()):
  """ Initialize Generator and Discriminator models """
  midi = MIDI(SEQ_LENGTH)
  generator = Generator()
  discriminator = Discriminator()

  if PRETRAINED:
    printf("Loading pretraining models")
    printf(DISGEN_PATH[-9:])
    preTrained = torch.load(DISGEN_PATH)
    printf("Last trained till epoch:", preTrained["epoch"])
    generator.load_state_dict(preTrained["g_model_state_dict"])
    discriminator.load_state_dict(preTrained["d_model_state_dict"])

  
  # Move models to specific device
  generator.to(device)
  discriminator.to(device)


  # Return models
  return generator, discriminator, midi


    
    
def make_directory_for_run():
  """ Make a directory for this training run. """
  printf(f'Preparing training run {UNIQUE_RUN_ID}')
  if not os.path.exists('./runs'):
    os.mkdir('./runs')
  os.mkdir(f'./runs/{UNIQUE_RUN_ID}')


# def decode_music(reconst_music, enCodecModel):
#   printf("im in decode music")
#   with torch.no_grad():
#     wav = enCodecModel.decode(reconst_music)

#   wav = wav.squeeze(0)

#   return wav


  
  
def print_training_progress(epoch, batch, generator_loss, discriminator_loss):
  """ Print training progress. """
  printf('Losses after mini-batch %5d: generator %e, discriminator %e' %
        (batch, generator_loss, discriminator_loss))
  wandb.log({'epoch': epoch, "batch":batch, 'GenLoss': generator_loss, 'DisLoss': discriminator_loss})


def prepare_dataset(midi):
  """ Prepare dataset through DataLoader """
  
  # TRAIN DATASET
  dataset = DLG(midi, PREPROCESSED_DATA_PATH)
  # Batch and shuffle data with DataLoader
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
  # Return dataset through DataLoader


  # TEST DATASET
#   testSet = DLG(tokenizer, bert_model, enCodecModel, mode="test")
  # Batch and shuffle data with DataLoader
#   testloader = torch.utils.data.DataLoader(testSet, batch_size=TEST_SET_SIZE, shuffle=False, num_workers=0, pin_memory=False)

  return trainloader



def initialize_loss():
  """ Initialize loss function. """
  return nn.BCELoss()

def initialize_optimizers(generator, discriminator):
  """ Initialize optimizers for Generator and Discriminator. """
  generator_optimizer = torch.optim.Adam(generator.parameters(), lr=OPTIMIZER_LR,betas=OPTIMIZER_BETAS)
  discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=OPTIMIZER_LR,betas=OPTIMIZER_BETAS)

  if PRETRAINED:
    printf("Loading Pretrained optimizers")
    preTrained = torch.load(DISGEN_PATH)
    generator_optimizer.load_state_dict(preTrained["g_optimizer_state_dict"])
    discriminator_optimizer.load_state_dict(preTrained["d_optimizer_state_dict"])
  return generator_optimizer, discriminator_optimizer
  




def efficient_zero_grad(model):
  """ 
    Apply zero_grad more efficiently
    Source: https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
  """
  for param in model.parameters():
    param.grad = None



def perform_train_step(batch, generator, discriminator, real_data, \
  loss_function, generator_optimizer, discriminator_optimizer, device = get_device()):
  """ Perform a single training step. """
  
  # 1. PREPARATION
  # Set real and fake labels.
  
  # Get images on CPU or GPU as configured and available
  # Also set 'actual baWWtch size', whih can be smaller than BATCH_SIZE
  # in some cases.
  real_batch_size = real_data.shape[0]
  fake_gen = torch.ones((real_batch_size, 1))   # swapped
  real = 0.1 * torch.rand_like(fake)        # 0-0.1
  fake = 0.1 * torch.rand_like(fake) + 0.9  # 0.9-1
  
  real_gen = torch.zeros((real_batch_size, 1))

  real_data = real_data.to(device)
  real = real.to(device)
  fake = fake.to(device)
  real_gen = real_gen.to(device)
  fake_gen = fake_gen.to(device)
  
  # Random noise for generator
  noise = generate_noise(real_batch_size)
  noise = noise.to(device)

  # Generate data using generator
  generator.eval()
  with torch.no_grad():
    gen_seqs = generator(noise)
  generator.train()


  # Train the discriminator
  if batch % TRAIN_1DISC_AFTER_BATCH == 0:
    efficient_zero_grad(discriminator)
    
    d_loss_real = loss_function(discriminator(real_data), real)
    d_loss_fake = loss_function(discriminator(gen_seqs.detach()), fake)

    d_loss = 0.5 * (d_loss_real + d_loss_fake)

    d_loss.backward()
    discriminator_optimizer.step()

  
  
  # TRAINING THE GENERATOR
  noise = generate_noise(real_batch_size)
  noise = noise.to(device)
  
  efficient_zero_grad(generator)
  pred = generator(noise)
    
  pred = discriminator(pred)

  g_loss = loss_function(pred, real_gen)
  g_loss.backward()
  generator_optimizer.step()
  
  return g_loss, d_loss
  

def perform_epoch(dataloader, generator, discriminator, loss_function, \
    generator_optimizer, discriminator_optimizer, epoch):
  """ Perform a single epoch. """
  global g_lossi_g, d_lossi_g
  g_lossi = d_lossi = []
  for batch_no, real_data in enumerate(dataloader, 0):
    # Perform training step
    printd(f'Starting Batch {batch_no} epoch {epoch}...')

    g_loss, d_loss = perform_train_step(batch_no, generator, \
      discriminator, real_data, loss_function, \
      generator_optimizer, discriminator_optimizer)
    # Print statistics and generate image after every n-th batch

    g_lossi.append(g_loss.item())
    d_lossi.append(d_loss.item())

    if batch_no % PRINT_STATS_AFTER_BATCH == 0:
      print_training_progress(epoch, batch_no, g_loss, d_loss)
      ### ADD GENERATE MUSIC VIA MIDI
    
    if batch_no%SAVE_MODEL_AFTER_BATCH == 0 and batch_no!= 0:
      save_models(generator, discriminator, generator_optimizer, discriminator_optimizer, epoch , g_lossi_g, d_lossi_g)

  torch.cuda.empty_cache()

  return np.mean(g_lossi), np.mean(d_lossi)



def save_models(generator, discriminator, generator_optimizer, discriminator_optimizer, epoch , g_loss, d_loss):
  print("saving")
  torch.save(
    {
    'epoch': epoch,
    'Run Name': UNIQUE_RUN_ID,
    'g_model_state_dict': generator.state_dict(),
    'g_optimizer_state_dict': generator_optimizer.state_dict(),
    'd_model_state_dict': discriminator.state_dict(),
    'd_optimizer_state_dict': discriminator_optimizer.state_dict(),
    "g_loss": g_loss,
    "d_loss": d_loss
    }, 
    f'./runs/{UNIQUE_RUN_ID}/cRnnGan.pth')
  print("saving done")
  

global g_lossi_g, d_lossi_g
g_lossi_g = d_lossi_g = []
def train_dcgan():
  # INIT WANDB
  wandb.init(project='GrooveGenie_crnnGan', entity='consolebot')

  



  global g_lossi_g, d_lossi_g
#   set_start_method('spawn')  # handling pyorch multiprocess cuda error
  """ Train the DCGAN. """
  # Make directory for unique run
  make_directory_for_run()
  
  # Set fixed random number seed
  torch.manual_seed(42)
  
  # Initialize models
  generator, discriminator, midi  = initialize_models()

  # Get prepared dataset
  trainloader = prepare_dataset(midi)
  
  #bert init
  # Initialize loss and optimizers
  loss_function = initialize_loss()
  generator_optimizer, discriminator_optimizer = initialize_optimizers(generator, discriminator)

  wandb.watch(generator, loss_function, log="all", log_freq= 5)
  wandb.watch(discriminator, loss_function, log="all", log_freq= 5)
  
  # Train the model
  

  if PRETRAINED:
    printf("Initializing previous losses track")
    preTrained = torch.load(DISGEN_PATH)
    printf("Previous avg Gen loss:", np.mean(preTrained["g_loss"]))
    printf("Previous avg Disc loss:", np.mean(preTrained["d_loss"]))
    g_lossi_g = preTrained["g_loss"]
    d_lossi_g = preTrained["d_loss"]

  for epoch in tqdm(range(NUM_EPOCHS)):
    print(f'Starting epoch {epoch}...')
    g_loss, d_loss = perform_epoch(trainloader, generator, discriminator, loss_function, \
      generator_optimizer, discriminator_optimizer, epoch)
    
    g_lossi_g.append(g_loss)
    d_lossi_g.append(d_loss)

    wandb.log({'epoch': epoch, 'EpochGenLoss': np.mean(g_lossi_g), 'EpochDisLoss': np.mean(d_lossi_g)})

    if epoch%SAVE_MODEL_AFTER_EPOCH == 0:
      save_models(generator, discriminator, generator_optimizer, discriminator_optimizer, epoch , g_lossi_g, d_lossi_g)

    if epoch%GENERATE_MUSIC_AFTER_EPOCH == 0:
      generate_music(generator, midi, epoch)

  # Finished :)
  printf(f'Finished unique run {UNIQUE_RUN_ID}')

