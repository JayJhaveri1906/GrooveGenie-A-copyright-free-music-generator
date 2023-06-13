import torch
from constants import *

def shape_for_decoding_codes():
  temp = [150] * TARGET_DURATION
  temp.append(TARGET_TOTAL-len(temp)*150)
  return temp 

def reverse_code(codes,scale):
    
    codes = codes.reshape((1,4,4242))
    arr_shape = shape_for_decoding_codes()
    
    # Initialize empty list to hold reconstructed encoded frames
    reconstructed_encoded_frames = []

    # Loop through each tensor in codes and split it into multiple tensors along the last dimension
    split_tensors = torch.split(codes, arr_shape, dim=-1)

    # Create tuples of split tensors and corresponding 2d-tensors from encoded_frames, and append to the reconstructed encoded frames list
    for i, split_tensor in enumerate(split_tensors):
        reconstructed_encoded_frames.append((split_tensor.detach(), scale[i].detach().reshape(1,1)))
      
    return reconstructed_encoded_frames