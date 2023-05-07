import uuid 
import torch


# Configurable variables
NUM_EPOCHS = 100
NOISE_DIMENSION = 50
BATCH_SIZE = 128
TRAIN_ON_GPU = True
UNIQUE_RUN_ID = str(uuid.uuid4())
PRINT_STATS_AFTER_BATCH = 100  # keep it above 62 to only generate once
OPTIMIZER_LR = 0.0002
OPTIMIZER_BETAS = (0.5, 0.999)
GENERATOR_OUTPUT_IMAGE_SHAPE = 28 * 28 * 1
TARGET_DURATION = 28 
TARGET_TOTAL = 4242
DEBUG = True
SAMPLE_RATE = 44100
TEST_SET_SIZE = 8
PRETRAINED = True
GEN_PATH = "runs/5d784cd3-92df-4719-9954-692a743260ac/generator_1_0.pth"
DIS_PATH = "runs/5d784cd3-92df-4719-9954-692a743260ac/discriminator_1_0.pth"


# common functions
def printf(*args):
  if DEBUG:
    print(" ".join(map(str,args)))


def get_device():
  """ Retrieve device based on settings and availability. """
  return torch.device("cuda:0" if torch.cuda.is_available() and TRAIN_ON_GPU else "cpu")