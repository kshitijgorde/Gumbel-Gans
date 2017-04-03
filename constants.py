# Dataset related
DATASET = 'oracle'

# Misc params
BATCH_SIZE = 32

# Pre-training
PRETRAIN_EPOCHS = 20
LEARNING_RATE_PRE_G = 0.001

# GAN
HIDDEN_STATE_SIZE = 32
HIDDEN_STATE_SIZE_D = HIDDEN_STATE_SIZE

LEARNING_RATE_G = 0.00005
LEARNING_RATE_D = LEARNING_RATE_G
N_EPOCHS = 200

NUM_D = 1
NUM_G = 3

######################## No configuration required ##############################
# Misc params
N_D_CLASSES = 2

# Dataset related
PG_VOCAB_SIZE = 5369
PG_SEQ_LENGTH = 21

PTB_VOCAB_SIZE = 50 # These are char level
PTB_SEQ_LENGTH = 492 # These are char level

ORACLE_VOCAB_SIZE = 5000
ORACLE_SEQ_LENGTH = 21
ORACLE_TEST_SIZE = 9984

# sanity checks
DATASET_LIST = ['ptb', 'pg', 'oracle']
assert DATASET in DATASET_LIST
assert ORACLE_TEST_SIZE%BATCH_SIZE == 0
assert NUM_G==1  or NUM_D==1

from utils import *

LOG_LOCATION = './logs/' + DATASET_LIST[:2] + '_g' + str(NUM_G) + 'd' + str(NUM_D) + '_g' + str(HIDDEN_STATE_SIZE) + '_d' + str(HIDDEN_STATE_SIZE_D) + '_pe' + str(PRETRAIN_EPOCHS) + '_pl' + str("{:.0e}".format(Decimal(LEARNING_RATE_PRE_G))) + '_l'+ str("{:.0e}".format(Decimal(LEARNING_RATE_G))) + '/'

PRETRAIN_CHK_FOLDER = './checkpoints/'  +  DATASET_LIST[:2] + '_p_h' + HIDDEN_STATE_SIZE + '_l' + str("{:.0e}".format(Decimal(LEARNING_RATE_PRE_G))) + '_e' + str(PRETRAIN_EPOCHS) + '/'
SAVE_FILE_PRETRAIN = PRETRAIN_CHK_FOLDER + DATASET_LIST[:2] + '_p_h' + HIDDEN_STATE_SIZE + '_l' + str("{:.0e}".format(Decimal(LEARNING_RATE_PRE_G))) + '.chk'
LOAD_FILE_PRETRAIN = SAVE_FILE_PRETRAIN

GAN_CHK_FOLDER = './checkpoints/' +  DATASET_LIST[:2] + '_g' + str(NUM_G) + 'd' + str(NUM_D) + '_g' + str(HIDDEN_STATE_SIZE) + '_d' + str(HIDDEN_STATE_SIZE_D) + '_pe' + str(PRETRAIN_EPOCHS) + '_pl' + str("{:.0e}".format(Decimal(LEARNING_RATE_PRE_G))) + '_l'+ str("{:.0e}".format(Decimal(LEARNING_RATE_G))) + '/'
SAVE_FILE_GAN = GAN_CHK_FOLDER + 'chk'
LOAD_FILE_GAN = SAVE_FILE_GAN

if SAVE_FILE_PRETRAIN:
	create_dir_if_not_exists('/'.join(SAVE_FILE_PRETRAIN.split('/')[:-1]))
if SAVE_FILE_GAN:
	create_dir_if_not_exists('/'.join(SAVE_FILE_GAN.split('/')[:-1]))
create_dir_if_not_exists(LOG_LOCATION)
