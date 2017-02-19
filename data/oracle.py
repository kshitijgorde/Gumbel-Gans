from MLE_SeqGAN.gen_dataloader import Gen_Data_loader, Likelihood_data_loader
from MLE_SeqGAN.pretrain_experiment import target_loss
from MLE_SeqGAN.target_lstm import TARGET_LSTM
from base import IterableDataset
import numpy as np
import os
import cPickle

# The methods in this file act as a convenient interface to the oracle code released by the authors of the SeqGAN paper

package_directory = os.path.dirname(os.path.abspath(__file__))
oracle_train_file = os.path.join(package_directory, 'MLE_SeqGAN/save/real_data.txt')
oracle_params_file = os.path.join(package_directory, 'MLE_SeqGAN/save/target_params.pkl')
oracle_eval_file = os.path.join(package_directory, 'MLE_SeqGAN/target_generate/gen_data.txt')

START_TOKEN = 0
VOCAB_SIZE = 5000

class OracleDataloader(IterableDataset):
    def __init__(self, batchsize, vocab_size, positive_file=oracle_train_file):
        IterableDataset.__init__(self)
        self.batchsize = batchsize
        self.vocab_size = vocab_size
        self.gen_data_loader = Gen_Data_loader(self.batchsize)
        self.gen_data_loader.create_batches(positive_file)
        self.start_char_idx = START_TOKEN

    def get_train_batch(self, _, one_hot=True):

        self.gen_data_loader.reset_pointer()

        for it in xrange(self.gen_data_loader.num_batch):
            batch = self.gen_data_loader.next_batch()

            if one_hot:
                yield self.one_hot(np.append([[START_TOKEN]]*self.batchsize, batch, axis=1))
            else:
                yield np.append([[START_TOKEN]]*self.batchsize, batch, axis=1)

    def get_mask(self):
        return super(OracleDataloader, self).get_mask(self.batchsize)

class OracleVerifier(object):
    """
    Sample code:
    from data.oracle import OracleVerifier as O
    import tensorflow as tf
    with tf.Session() as sess:
        o=O(32, sess)
        tf.initialize_all_variables().run()

        # generate sample file

        print o.get_loss(sample_file)
    """
    def __init__(self, batchsize, session, eval_file=oracle_eval_file):
        self.batchsize = batchsize
        target_params = cPickle.load(open(oracle_params_file))
        self.target_lstm = TARGET_LSTM(VOCAB_SIZE, 32, 32, 32, 20, 0, target_params)
        self.likelihood_data_loader = Likelihood_data_loader(self.batchsize)
        self.sess = session
        self.eval_file = oracle_eval_file

    def get_loss(self, eval_file=None):
        if not eval_file:
            eval_file = self.eval_file
        self.likelihood_data_loader.create_batches(eval_file)
        test_loss = target_loss(self.sess, self.target_lstm, self.likelihood_data_loader)
        return test_loss
