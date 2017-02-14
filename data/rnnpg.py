import numpy as np
import os
from base import IterableDataset

package_directory = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(package_directory, 'rnnpg/pgdata')

"""
Example code for character-level prediction:

    c = CharLevelRNNPG()
    for b in c.get_train_batch(20):
        # do stuff on batch b
"""
class CharLevelRNNPG(IterableDataset):
    def __init__(self):
        IterableDataset.__init__(self)
        self.toks = [ 'S', 'E']
        self.start_char_idx = 0
        self.char2idx = {}
        self.idx2char = []
        self.sent_length = 20
        self.line_count = 0

        self.load_stats()
        self.load_data()

    def load_stats(self):
        for t in self.toks:
            self.char2idx[t] = len(self.char2idx)
            self.idx2char += [t]

        with open(fname, 'r') as f:
            for l in f:
                self.line_count += 1
                l = l.strip().replace(' ', '').replace('\t', '')
                assert len(l)%3 == 0 and len(l)/3 == self.sent_length

                # get chars triplet-wise: http://stackoverflow.com/a/18121510
                chars = [str(l[::-1][i:i+3][::-1]) for i in range(0, len(l), 3)][::-1]
                for c in chars:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.char2idx)
                        self.idx2char += [c]

        self.sent_length += 2 # since there's always an sos and eos token
        self.vocab_size = len(self.idx2char)

    def load_data(self):
        self.train = np.zeros((self.line_count, self.sent_length), dtype='uint16')

        self.train[:, 0] = self.char2idx['S']
        self.train[:, -1] = self.char2idx['E']
        with open(fname, 'r') as f:
            for ln, l in enumerate(f):
                l = l.strip().replace(' ', '').replace('\t', '')
                chars = [str(l[::-1][i:i+3][::-1]) for i in range(0, len(l), 3)][::-1]
                # print l,ln
                for j, w in enumerate(chars):
                    # print j,w
                    self.train[ln, j+1] = self.char2idx[w]


    def get_train_batch(self, batch, **kwargs):
        if 'max_size' not in kwargs:
            kwargs['max_size'] = 16394
        return super(CharLevelRNNPG, self).get_train_batch(batch, **kwargs)
