PG_VOCAB_SIZE = 5369
PG_SEQ_LENGTH = 21

# These are char level
PTB_VOCAB_SIZE = 50
PTB_SEQ_LENGTH = 492

ORACLE_VOCAB_SIZE = 5000
ORACLE_SEQ_LENGTH = 21
ORACLE_TEST_SIZE = 9984

DATASET = 'oracle'

HIDDEN_STATE_SIZE = 32
HIDDEN_STATE_SIZE_D = HIDDEN_STATE_SIZE

BATCH_SIZE = 32
N_CLASSES = 2
LEARNING_RATE_G = 0.00005
LEARNING_RATE_D = 0.00005
N_EPOCHS = 200

PRETRAIN_EPOCHS = 20
LEARNING_RATE_PRE_G = 0.001

PRETRAIN_CHK_FOLDER = './checkpoints/or_p_h32_le-3_e20/'
SAVE_FILE_PRETRAIN = PRETRAIN_CHK_FOLDER + 'or_p_h32_le-3.chk'
LOAD_FILE_PRETRAIN = SAVE_FILE_PRETRAIN
NUM_D = 1
NUM_G = 3

GAN_CHK_FOLDER = './checkpoints/or_g32_d32_l5e-5/'
SAVE_FILE_GAN = GAN_CHK_FOLDER + 'chk'
LOAD_FILE_GAN = SAVE_FILE_GAN

LOG_LOCATION = './logs/pg_g' + str(NUM_G) + 'd' + str(NUM_D) + '_g512_d512_p0_l5e-5/'

# sanity checks:
DATASET_LIST = ['ptb', 'pg', 'oracle']
assert DATASET in DATASET_LIST
assert ORACLE_TEST_SIZE%BATCH_SIZE == 0
assert NUM_G==1  or NUM_D==1

from utils import *

if SAVE_FILE_PRETRAIN:
	create_dir_if_not_exists('/'.join(SAVE_FILE_PRETRAIN.split('/')[:-1]))
if SAVE_FILE_GAN:
	create_dir_if_not_exists('/'.join(SAVE_FILE_GAN.split('/')[:-1]))
create_dir_if_not_exists(LOG_LOCATION)
