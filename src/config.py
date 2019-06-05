import numpy as np

# DATASET PARAMETERS
TRAIN_DIR = '/datasets/nyud-with-depth/'  ### old: /datasets/nyud/
VAL_DIR = TRAIN_DIR
TRAIN_LIST = ['./data/training.txt'] * 3  ### old: ./data/train.nyu
VAL_LIST = ['./data/testing.txt'] * 3  ### old: ./data/val.nyu
SHORTER_SIDE = [350] * 3
CROP_SIZE = [500] * 3
NORMALISE_PARAMS = [1./255, # SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)), # MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)), # STD
                    1./255, # depth scale
                    np.array([0.440, 0.440, 0.520]).reshape((1, 1, 3)), # depth mean
                    np.array([0.15, 0.270, 0.230]).reshape((1, 1, 3))] # depth std
BATCH_SIZE = [3] * 3   ### max without running out of GPU memory for K80
NUM_WORKERS = 16
NUM_CLASSES = [41] * 3  # 40 + void
LOW_SCALE = [0.5] * 3
HIGH_SCALE = [2.0] * 3
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = '50'
ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised 

# GENERAL
EVALUATE = False
FREEZE_BN = [True] * 3
NUM_SEGM_EPOCHS = [100] * 3
PRINT_EVERY = 50  #10
RANDOM_SEED = 42
SNAPSHOT_DIR = './ckpt/'
CKPT_PATH = './ckpt/checkpoint.pth.tar'
VAL_EVERY = [5] * 3 # how often to record validation scores

# OPTIMISERS' PARAMETERS
LR_ENC = [5e-4, 2.5e-4, 1e-4]  # TO FREEZE, PUT 0
LR_DEC = [5e-3, 2.5e-3, 1e-3]  # 1e-4 for adam
MOM_ENC = [0.9] * 3 # TO FREEZE, PUT 0
MOM_DEC = [0.9] * 3
WD_ENC = [1e-5] * 3 # TO FREEZE, PUT 0
WD_DEC = [1e-5] * 3
OPTIM_DEC = 'sgd'
