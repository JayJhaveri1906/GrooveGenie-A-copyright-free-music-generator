from ulid import ULID
import torch


# MIDI 
DEBUG = False
VERBOSE = True

SEQ_LENGTH = 100
SEQ_SHAPE = SEQ_LENGTH
LATENT_DIM = 1000

PREPROCESSED_DATA_PATH = "DataNew/PreProcessed/LakhDSOneSong.pth"


# Configurable variables
NUM_EPOCHS = 101
NOISE_DIMENSION = 50
BATCH_SIZE = 128
TRAIN_ON_GPU = True
UNIQUE_RUN_ID = str(ULID().milliseconds)
PRINT_STATS_AFTER_BATCH = 50
SAVE_MODEL_AFTER_BATCH = 5000
SAVE_MODEL_AFTER_EPOCH = 10
GENERATE_MUSIC_AFTER_EPOCH = 20
OPTIMIZER_LR = 0.0002
OPTIMIZER_BETAS = (0.5, 0.999)
GENERATOR_OUTPUT_IMAGE_SHAPE = 28 * 28 * 1
TARGET_DURATION = 28 
TARGET_TOTAL = 4242
SAMPLE_RATE = 44100
TEST_SET_SIZE = 8
PRETRAINED = False
DISGEN_PATH = "runs/1685129681226/cRnnGan.pth"
TRAIN_1DISC_AFTER_BATCH = 50


# common functions
def printf(*args):
  if VERBOSE:
    print(" ".join(map(str,args)))

def printd(*args):
  if DEBUG:
    print(" ".join(map(str,args)))


def get_device():
  """ Retrieve device based on settings and availability. """
  return torch.device("cuda:0" if torch.cuda.is_available() and TRAIN_ON_GPU else "cpu")