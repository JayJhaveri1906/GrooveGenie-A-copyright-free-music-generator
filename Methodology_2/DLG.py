from torch.utils import data
from constants import *
import sys
from preprocessing import load_dict_from_file


class DLG(data.Dataset):
  def __init__(self, midi, data="DataNew/PreProcessed/LakhDS1_.pth", mode="train"):
    try:
      transfer_dic, self.train_sequences = load_dict_from_file(data)
      midi.transfer_dic = transfer_dic
    except:
      sys.exit("Preprocessed file not found. Please use preprocessing.py to preprocess your data.")


  def __getitem__(self, index):
    return self.train_sequences[index]
  

  def __len__(self):
    return len(self.train_sequences)