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
        self.generate_data()
        self.generate_valid_test()

    def generate_data(self):
        self.train_list = []
        self.valid_list = []
        self.test_list = []

        file_mapping = {}
        file_mapping[files[0]] = self.valid_list
        file_mapping[files[1]] = self.train_list
        file_mapping[files[2]] = self.test_list

        for fname in files:
            with open(fname, 'r') as f:
                for l in f:
                    l = l.strip().replace('<unk>', 'U')
                    if self.use_new_lines:
                        l += '\n'
                    chars = list(l)
                    file_mapping[fname] += [self.char2idx[w] for w in chars]

        self.train_list = np.asarray(self.train_list, dtype='uint16')
        self.valid_list = np.asarray(self.valid_list, dtype='uint16')
        self.test_list = np.asarray(self.test_list, dtype='uint16')

    def generate_valid_test(self):
        num_batches = (len(self.valid_list) // self.seqlen)
        self.valid = self.valid_list[:num_batches*self.seqlen].reshape((-1, self.seqlen))

        num_batches = (len(self.test_list) // self.seqlen)
        self.test = self.test_list[:num_batches*self.seqlen].reshape((-1, self.seqlen))

    def test_onehot_to_string(self, onehot):
        '''
        given the one-hot encoded dataset, return the text string
        '''
        return ''.join([self.idx2char[i] for i in np.where(onehot[0,:])[-1]])

    def get_train_batch(self, batch, augment=True, **kwargs):
        if augment:
            s_ind = np.random.randint(0, self.seqlen)
        else:
            s_ind = 0
        num_batches = ((len(self.train_list)-s_ind) // self.seqlen)
        self.train = self.train_list[s_ind:s_ind+num_batches*self.seqlen].reshape((-1, self.seqlen))

        for b in super(CharLevelPTB, self).get_train_batch(batch, **kwargs):
            yield b

    def get_valid_batch(self, batch, **kwargs):
        for b in super(CharLevelPTB, self).get_valid_batch(batch, **kwargs):
            yield b

    def get_test_batch(self, batch, **kwargs):
        for b in super(CharLevelPTB, self).get_test_batch(batch, **kwargs):
            yield b
