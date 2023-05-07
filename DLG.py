import os
from torch.utils import data
import torch
import numpy as np
import pandas as pd
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
from constants import *
import torch

## OLD CODE
def read_encode_audio(music_path, model, ext=".mp3" , device = get_device()):
  printf("im in read encode audio")
  # Load and pre-process the audio waveform
  wav, sr = torchaudio.load("Data/Music/small_songs/"+music_path+ext)

  target_duration = 28
  #target_sr = model.sample_rate ## doesn't work good
  target_length = int(target_duration * sr)

  # Create Resample transformation
  resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sr, 
                                                      resampling_method='sinc_interpolation')

  # Apply transformation to audio waveform
  wav = resample_transform(wav[:, :target_length])

  wav = convert_audio(wav, sr, model.sample_rate, model.channels)
  wav = wav.unsqueeze(0)
  
  # Extract discrete codes from EnCodec
  with torch.no_grad():
      encoded_frames = model.encode(wav.to(device))
    
  del wav
  codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1) # [1, 4, 4242]
  codes = codes.reshape((4 * 4242))  # 16968
  scales = torch.cat([encoded[1] for encoded in encoded_frames], dim=-1).squeeze(0) # 29

  return codes, scales


def get_embedding(genre, tokenizer, model, device = get_device()):
  printf("im in get embedding")

  def makeinput(listi):
    # stri= "I want a song from the following genres: "
    stri = ""
    for elem in listi:
        stri += elem + " "
    return stri[:-1]

  # Load the pre-trained BERT model and tokenizer
  # Get the input sentence from the makeinput function
  input_sentence = makeinput(genre)

  # Tokenize the input sentence
  input_tokens = tokenizer.encode(input_sentence, add_special_tokens=False)

  # Convert tokens to a PyTorch tensor
  input_tensor = torch.tensor([input_tokens])

  # Pass the input tensor through the BERT model to get the embedding
  with torch.no_grad():
    embedding = model(input_tensor.to(device))[0][0]
  
  return torch.mean(embedding, axis = 0)



def read_preprocessed(var, track_id, device = get_device()):
  path = f'./Data/Preprocessed/{var}/{track_id}.pt'
  tensor = torch.load(path)
  
  return tensor.to(device)




class DLG(data.Dataset):
  def __init__(self, bert_tokenizer, bert_model, enCodecModel, mode):

    self.mode = mode
    if mode == "train":
      self.music_genre = pd.read_excel("Data/small_dataset.xlsx", dtype={"track_id":str})
      self.music_genre["track_id"] = self.music_genre["track_id"].astype(str).str.zfill(6)

      self.music_genre = self.music_genre.drop(columns=["dataset"])
      if len(self.music_genre) == 0:
        raise RuntimeError('Found 0 music, please check the data set')

    elif mode == "test":
      self.genres = [{'Hip-Hop'},
                {'Pop'},
                {'Folk'},
                {'Experimental'},
                {'Rock'},
                {'International'},
                {'Electronic'},
                {'Instrumental'}]
      self.bert_model = bert_model
      self.bert_tokenizer = bert_tokenizer
    
    else:
      print("INVALID MODE")
      assert 1!=1
      
      

  def __getitem__(self, index):

    if self.mode == "train":
      track_id, genre = self.music_genre.iloc[index]

      codes = read_preprocessed("codes", track_id)
      scale = read_preprocessed("scales", track_id)
      genre = read_preprocessed("genres", track_id)

      # Old Code
      # codes, scale = read_encode_audio(music,self.enCodecModel)
      # genre = get_embedding(genre, self.bert_tokenizer, self.bert_model)

      return codes, scale, genre

    elif self.mode == "test":
      genre = self.genres[index]
      genre = get_embedding(genre, self.bert_tokenizer, self.bert_model)
      return genre

  def __len__(self):
    if self.mode == "train":
      return len(self.music_genre)
    elif self.mode == "test":
      return len(self.genres)