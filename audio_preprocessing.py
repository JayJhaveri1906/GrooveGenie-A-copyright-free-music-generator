import pandas as pd
from transformers import BertTokenizer, BertModel
from encodec import EncodecModel
import torchaudio
from encodec.utils import convert_audio
import torch
from constants import *
import os

# songs_genre = pd.read_excel("Data/Music/small_dataset.xlsx")

def get_device():
  """ Retrieve device based on settings and availability. """
  return torch.device("cuda:0" if torch.cuda.is_available() and TRAIN_ON_GPU else "cpu")

def initialize_Bert():
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')
  return tokenizer,model

def initialize_EnCodec():
  model = EncodecModel.encodec_model_48khz()
  model.set_target_bandwidth(6.0)
  return model

def read_encode_audio(music_path, model, ext=".mp3" , device = get_device()):
  wav, sr = torchaudio.load("Data/Music/small_songs/"+music_path+ext)

  target_duration = 28
  target_length = int(target_duration * sr)
  resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sr, 
                                                      resampling_method='sinc_interpolation')
  wav = resample_transform(wav[:, :target_length])

  wav = convert_audio(wav, sr, model.sample_rate, model.channels)
  wav = wav.unsqueeze(0)
  
  with torch.no_grad():
      encoded_frames = model.encode(wav.to(device))
  
  del wav
  codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1) # [1, 4, 4242]
  codes = codes.reshape((4 * 4242))  # 16968
  scales = torch.cat([encoded[1] for encoded in encoded_frames], dim=-1).squeeze(0) # 29

  return codes , scales


def get_embedding(genre, tokenizer, model, device = get_device()):
  def makeinput(listi):
    stri = ""
    for elem in listi:
        stri += elem + " "
    return stri[:-1]

  input_sentence = makeinput(genre)
  input_tokens = tokenizer.encode(input_sentence, add_special_tokens=False)
  input_tensor = torch.tensor([input_tokens])

  with torch.no_grad():
    embedding = model(input_tensor.to(device))[0][0]
  
  return torch.mean(embedding, axis = 0)

if __name__ == '__main__':
  if not os.path.exists(f'./Data/Preprocessed/codes/'):
    os.mkdir(f'./Data/Preprocessed/codes/')

  if not os.path.exists(f'./Data/Preprocessed/scales/'):
    os.mkdir(f'./Data/Preprocessed/scales/')

  if not os.path.exists(f'./Data/Preprocessed/genres/'):
    os.mkdir(f'./Data/Preprocessed/genres/')

  tokenizer, bert_model = initialize_Bert()
  enCodecModel = initialize_EnCodec()
  
  device = get_device()
  bert_model.to(device)
  enCodecModel.to(device)

  music_genre = pd.read_excel("Data/small_dataset.xlsx", dtype={"track_id":str})
  music_genre["track_id"] = music_genre["track_id"].astype(str).str.zfill(6)
  music_genre = music_genre.drop(columns=["dataset"])

  for i in range(len(music_genre)):
    music, genre = music_genre.iloc[i]
    codes, scale = read_encode_audio(music, enCodecModel)
    genre = get_embedding(genre, tokenizer, bert_model)

    ## save
    torch.save(codes, f'./Data/Preprocessed/codes/{music}.pt')
    torch.save(scale, f'./Data/Preprocessed/scales/{music}.pt')
    torch.save(genre, f'./Data/Preprocessed/genres/{music}.pt')

    if i % 100:
      print(i,music)





