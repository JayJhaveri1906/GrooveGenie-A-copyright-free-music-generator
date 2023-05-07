from utils import train_dcgan
import torch
from constants import get_device


# Speed ups
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
  print("STARTING")
  print("Running on", get_device())
  train_dcgan()
