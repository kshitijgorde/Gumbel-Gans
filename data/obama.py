import numpy as np
import os
from base import IterableDataset
import codecs

try:
    xrange
except NameError:
    xrange = range

package_directory = os.path.dirname(os.path.abspath(__file__))
files = [os.path.join(package_directory, datafile) for datafile in ['obama/input.txt']]

"""
Example code for character-level prediction:

    c = CharLevelObama()
    for b in c.get_train_batch(20):
        # do stuff on batch b
"""

class CharLevelObama(IterableDataset):
    def __init__(self, seqlen=100, split=0.8, use_new_lines=True, tag_end_speech=True):
        IterableDataset.__init__(self)
        self.use_new_lines = use_new_lines
        self.tag_end_speech = tag_end_speech
        self.toks = []
        if self.use_new_lines:
            self.toks += ['\n']
        if self.tag_end_speech:
            self.toks += ['@']
        self.char2idx = {}
        self.idx2char = []
        self.seqlen = seqlen
        self.split = split
        self.data = []

        self.load_stats()
        self.load_data()

    def load_stats(self):
        for t in self.toks:
            self.char2idx[t] = len(self.char2idx)
            self.idx2char += [t]

        for fname in files:
            with codecs.open(fname, encoding='utf-8') as f:
                for l in f:
                    l = l.strip()
                    chars = list(l)
                    for c in chars:
                        if c not in self.char2idx:
                            self.char2idx[c] = len(self.char2idx)
                            self.idx2char += [c]

        self.vocab_size = len(self.idx2char)

    def load_data(self):
        self.generate_data()
        # self.generate_valid_test()

    def generate_data(self):
        self.train_list = []
        self.test_list = []

        new_line_count = 0
        for fname in files:
            with codecs.open(fname, encoding='utf-8') as f:
                for l in f:
                    l = " ".join(l.strip().split())
                    if l == '':
                        new_line_count += 1
                    else:
                        if new_line_count <= 3:
                            if self.use_new_lines:
                                l = '\n' + l
                            else:
                                l = ' ' + l
                        elif new_line_count > 3:
                            if self.tag_end_speech:
                                l = '@' + l
                            elif self.use_new_lines:
                                l = '\n' + l
                            else:
                                l = ' ' + l
                        new_line_count = 0
                        self.data += list(l)

        self.data = self.data[1:] # remove spurious newline at start of sentence

        train_test_split = ''.join(self.data).split('@')
        pivot = int(len(train_test_split)*self.split)

        self.train_list = np.asarray([self.char2idx[w] for w in '@'.join(train_test_split[:pivot])], dtype='uint16')
        self.test_list = np.asarray([self.char2idx[w] for w in '@'.join(train_test_split[pivot:])], dtype='uint16')

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

        for b in super(CharLevelObama, self).get_train_batch(batch, **kwargs):
            yield b

    def get_valid_batch(self, batch, **kwargs):
        raise NotImplementedError

    def get_test_batch(self, batch, **kwargs):
        raise NotImplementedError
