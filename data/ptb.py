import numpy as np
import os
from base import IterableDataset

try:
    xrange
except NameError:
    xrange = range

package_directory = os.path.dirname(os.path.abspath(__file__))
files = [os.path.join(package_directory, datafile) for datafile in ['ptb/valid.txt', 'ptb/train.txt', 'ptb/test.txt']]

"""
Example code for character-level prediction:

    c = CharLevelPTB()
    for b in c.get_train_batch(20):
        # do stuff on batch b
"""
class WordLevelPTB(IterableDataset):
    def __init__(self):
        IterableDataset.__init__(self)
        self.toks = [ '<zeropad>', '<sos>', '<eos>', '<unk>']
        self.word2idx = {}
        self.idx2word = []
        self.max_sent_length = 0
        self.line_lengths = {}

        self.load_stats()
        self.load_data()


    def load_stats(self):
        for t in self.toks:
            self.word2idx[t] = len(self.word2idx)
            self.idx2word += [t]

        for fname in files:
            with open(fname, 'r') as f:
                self.line_lengths[fname] = 0
                for l in f:
                    self.line_lengths[fname] += 1
                    words = l.split()
                    self.max_sent_length = max(len(words), self.max_sent_length)
                    for w in words:
                        if w not in self.word2idx:
                            self.word2idx[w] = len(self.word2idx)
                            self.idx2word += [w]

        self.max_sent_length += 2 # since there's always an sos and eos token
        self.vocab_size = len(self.idx2word)


    def load_data(self):
        self.train = np.zeros((self.line_lengths[files[1]], self.max_sent_length), dtype='uint16')
        self.valid = np.zeros((self.line_lengths[files[0]], self.max_sent_length), dtype='uint16')
        self.test = np.zeros((self.line_lengths[files[2]], self.max_sent_length), dtype='uint16')

        file_mapping = {}
        file_mapping[files[0]] = self.valid
        file_mapping[files[1]] = self.train
        file_mapping[files[2]] = self.test

        for fname in files:
            file_mapping[fname][:, 0] = self.word2idx['<sos>']
            with open(fname, 'r') as f:
                for i, l in enumerate(f):
                    words = l.split()
                    for j, w in enumerate(words):
                        file_mapping[fname][i, j+1] = self.word2idx[w]
                    file_mapping[fname][i, len(words)+1] = self.word2idx['<eos>']

class CharLevelPTB(IterableDataset):
    TRAIN_FIRST_CHARS = 'aer banknote berlitz calloway centrust cluett fromstein gitano guterman hydro-quebec ipo kia memotec'
    VAL_FIRST_CHARS = 'consumers may want to move their telephones a little closer to the tv set\nU U watching abc \'s monday night football can now vote during'
    TEST_FIRST_CHARS = 'no it was n\'t black monday\nbut while the new york stock exchange did n\'t fall apart friday as the dow jones industrial'
    def __init__(self, seqlen=100, use_new_lines=True, file_location=os.path.join(package_directory, 'ptb/char_ptb.npz')):
        IterableDataset.__init__(self)
        self.use_new_lines = use_new_lines
        self.toks = ['a', 'U']
        if self.use_new_lines:
            self.toks += ['\n']
        self.char2idx = {}
        self.idx2char = []
        self.seqlen = seqlen
        self.file_location = file_location

        self.load_stats()
        self.load_data()

    def load_stats(self):
        for t in self.toks:
            self.char2idx[t] = len(self.char2idx)
            self.idx2char += [t]

        for fname in files:
            with open(fname, 'r') as f:
                for l in f:
                    l = l.strip().replace('<unk>', 'U')
                    chars = list(l)
                    for c in chars:
                        if c not in self.char2idx:
                            self.char2idx[c] = len(self.char2idx)
                            self.idx2char += [c]

        self.vocab_size = len(self.idx2char)

    def load_data(self):
        if  os.path.exists(self.file_location):
            self.load_from_file()
        else:
            self.generate_data()
            self.save_data()
        self.test_onehot_to_string(self.train, self.TRAIN_FIRST_CHARS[:self.seqlen])
        self.test_onehot_to_string(self.valid, self.VAL_FIRST_CHARS[:self.seqlen])
        self.test_onehot_to_string(self.test, self.TEST_FIRST_CHARS[:self.seqlen])

    def generate_data(self):
        train_list = []
        valid_list = []
        test_list = []

        file_mapping = {}
        file_mapping[files[0]] = valid_list
        file_mapping[files[1]] = train_list
        file_mapping[files[2]] = test_list

        # import pdb

        for fname in files:
            with open(fname, 'r') as f:
                for l in f:
                    l = l.strip().replace('<unk>', 'U')
                    if self.use_new_lines:
                        l += '\n'
                    chars = list(l)
                    file_mapping[fname] += [self.char2idx[w] for w in chars]

        train_list = np.asarray(train_list, dtype='uint16')
        valid_list = np.asarray(valid_list, dtype='uint16')
        test_list = np.asarray(test_list, dtype='uint16')

        def generate_sequences(dest, src):
            for i in xrange(dest.shape[0]):
                dest[i] = self.one_hot(src[i:i+dest.shape[1]])

        self.train = np.zeros((len(train_list)-self.seqlen+1, self.seqlen, self.vocab_size), dtype='uint16')
        generate_sequences(self.train, train_list)
        self.valid = np.zeros((len(valid_list)-self.seqlen+1, self.seqlen, self.vocab_size), dtype='uint16')
        generate_sequences(self.valid, valid_list)
        self.test = np.zeros((len(test_list)-self.seqlen+1, self.seqlen, self.vocab_size), dtype='uint16')
        generate_sequences(self.test, test_list)
        # pdb.set_trace()

    def save_data(self):
        np.savez(self.file_location, train=self.train, valid=self.valid, test=self.test)

    def load_from_file(self):
        try:
            a = np.load(self.file_location)
        except MemoryError:
            a = np.load(self.file_location, mmap_mode='r')
        self.train = a['train']
        self.valid = a['valid']
        self.test = a['test']

    def test_onehot_to_string(self, onehot, string):
        '''
        given the one-hot encoded dataset and a string, ensures the first sequence matches the string
        '''
        assert ''.join([self.idx2char[i] for i in np.where(onehot[0,:])[-1]]) == string

    def get_train_batch(self, batch, **kwargs):
        kwargs['one_hot'] = False
        return super(CharLevelRNNPG, self).get_train_batch(batch, **kwargs)

    def get_valid_batch(self, batch, **kwargs):
        kwargs['one_hot'] = False
        return super(CharLevelRNNPG, self).get_valid_batch(batch, **kwargs)

    def get_test_batch(self, batch, **kwargs):
        kwargs['one_hot'] = False
        return super(CharLevelRNNPG, self).get_test_batch(batch, **kwargs)

    # def window_stack(self, a, num_windows=3, stepsize=1):
    #     # I<3SO http://stackoverflow.com/a/15722507
    #     n = a.shape[0]
    #     return np.hstack( a[i:1+n+i-num_windows:stepsize] for i in range(0,num_windows) )

    # def store_onehot_sequence(self, dest, index, numd_string):
    #     for j in range(len(dest)):
    #         dest[j] = self.one_hot(numd_string[index+j:index+j+self.seqlen])

    # def get_train_batch(self, batchsize):
    #     tr = np.zeros((batchsize, self.seqlen, self.vocab_size), dtype='uint16')
    #     for i in xrange(0, len(self.train_list)-self.seqlen - batchsize + 1, batchsize):
    #         w = self.window_stack(self.train_list[i:i+batchsize+self.seqlen-1], batchsize).reshape((batchsize, self.seqlen))
    #         yield self.one_hot(w)

    # def get_valid_batch(self, batchsize):
    #     va = np.zeros((batchsize, self.seqlen, self.vocab_size), dtype='uint16')
    #     for i in xrange(0, len(self.valid_list)-self.seqlen - batchsize + 1, batchsize):
    #         self.store_onehot_sequence(va, i, self.valid_list)
    #         yield va

    # def get_test_batch(self, batchsize):
    #     te = np.zeros((batchsize, self.seqlen, self.vocab_size), dtype='uint16')
    #     for i in xrange(0, len(self.test_list)-self.seqlen - batchsize + 1, batchsize):
    #         self.store_onehot_sequence(te, i, self.test_list)
    #         yield te

