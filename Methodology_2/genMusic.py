from constants import *
import torch
import os
from midiHandler import MIDI
from music21 import converter
from lstmGan import Generator
import numpy as np
# from midi2audio import FluidSynth


def convertMidiToWAV(midi_path, save_path = "output.wav", sound_font_path = "GeneralUser GS v1.471.sf2"):
  # save_path must end with flac or wav if FluidSynth
  # fs = FluidSynth(sound_font_path)
  # fs.midi_to_audio(midi_path, save_path)
  midi_ = converter.parse(midi_path)
  midi_.write('wav', save_path)
  

def generate_noise(real_batch_size = BATCH_SIZE, latent_dim = LATENT_DIM, device=get_device()):
  """ Generate noise music: according to batch size and LatentDim"""
  noise = torch.randn((real_batch_size, latent_dim)).to(device) 
  return noise


def generate_music(generator, midi, epoch = 0, noOfSamples=1, UNIQUE_RUN_ID = UNIQUE_RUN_ID, device=get_device()):
  """ Generate subplots with generated examples. """
  printf("Generating")
  
  generator.eval()

  for i in range(noOfSamples):
    noise = generate_noise(1)
    
    with torch.no_grad():
      gen_seq = generator(noise)


    # transfer sequence numbers to notes
    boundary = int(len(midi.transfer_dic) / 2)
    pred_nums = [x * boundary + boundary for x in gen_seq[0]]
    notes = [key for key in midi.transfer_dic]
    values = [tensor.item() for tensor in pred_nums]
    minimum_value = min(values)
    maximum_value = max(values)
    print(minimum_value, maximum_value)
    print(len(notes))
    pred_notes = [notes[min(max(int(x.item()),0),len(notes)-1)] for x in pred_nums]


    if not os.path.exists(f'./runs/{UNIQUE_RUN_ID}/music'):
      os.mkdir(f'./runs/{UNIQUE_RUN_ID}/music')

    if not os.path.exists(f'./runs/{UNIQUE_RUN_ID}/music/ep{epoch}'):
      os.mkdir(f'./runs/{UNIQUE_RUN_ID}/music/ep{epoch}')

    fname = f'./runs/{UNIQUE_RUN_ID}/music/ep{epoch}/epoch{epoch}_{i}'

    midi.create_midi(pred_notes, fname)

  printf("Generating Done")
  generator.train()

if __name__ == "__main__":
  midi = MIDI(SEQ_LENGTH)
  generator = Generator()
  device = get_device()

  midiLoader = torch.load(PREPROCESSED_DATA_PATH)
  midi.transfer_dic = midiLoader["transfer_dic"]

  printf("Loading pretraining models")
  printf(DISGEN_PATH[-9:])
  preTrained = torch.load(DISGEN_PATH)
  printf("Last trained till epoch:", preTrained["epoch"])
  printf("Previous avg Gen loss:", np.mean(preTrained["g_loss"]))
  printf("Previous avg Disc loss:", np.mean(preTrained["d_loss"]))
  generator.load_state_dict(preTrained["g_model_state_dict"])
                          
  generator.to(device)

  generate_music(generator, midi, epoch = "INFERENCE", noOfSamples=10, UNIQUE_RUN_ID=preTrained["Run Name"])
