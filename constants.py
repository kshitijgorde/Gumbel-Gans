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