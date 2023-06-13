from constants import *
from midiHandler import MIDI
import pickle

def preprocesser(data, preprocess):
  midi = MIDI(SEQ_LENGTH)

  printf("DLG INIT")
  
  printf("Parsing midis")
  midi.parser(data)
  transfer_dic = midi.transfer_dic

  printf("in prepare_sequences")
  train_sequences = midi.prepare_sequences()
  printf(f"\nNumber of sequences for train: {train_sequences.shape[0]}\n")

  train_sequences = torch.from_numpy(train_sequences).float()

  save_dict_to_file(train_sequences, transfer_dic, preprocess)




def save_dict_to_file(train_sequences, transfer_dic, file_path):
  print("saving")
  torch.save(
    {
    'train_sequences': train_sequences,
    'transfer_dic': transfer_dic
    }, 
    file_path)
  print("saving done")


#   with open(file_path, 'wb') as file:
#     pickle.dump(dictionary, file)

def load_dict_from_file(file_path):
  preTrained = torch.load(file_path)
  train_sequences = preTrained["train_sequences"]
  transfer_dic = preTrained["transfer_dic"]

  return transfer_dic, train_sequences
#   with open(file_path, 'rb') as file:
#     return pickle.load(file)


if __name__ == "__main__":
  
  # for i in range(1,10):
  dataPath = "DataNew/train_oneSong"
  preprocessPath = "DataNew/PreProcessed/cutsLikeAKnife.pth"
  preprocesser(dataPath, preprocessPath)
