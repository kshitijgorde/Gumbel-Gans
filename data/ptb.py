import numpy as np
import os

package_directory = os.path.dirname(os.path.abspath(__file__))
files = [os.path.join(package_directory, datafile) for datafile in ['ptb/valid.txt', 'ptb/train.txt', 'ptb/test.txt']]

class IterableDataset:
    def __init__(self):
        self.train = np.zeros((0, 0))
        self.valid = np.zeros((0, 0))
        self.test = np.zeros((0, 0))

    def get_train_batch(self, batchsize, use_remainder_batch=False, shuffle=True):
        indices = np.arange(len(self.train))

        if shuffle:
            np.random.shuffle(indices)

        if use_remainder_batch:
            for i in range(0, len(self.train), batchsize):
                yield self.train[indices[i:i+batchsize]]
        else:
            for i in range(0, len(self.train) - batchsize + 1, batchsize):
                yield self.train[indices[i:i+batchsize]]

    def get_valid_batch(self, batchsize, use_remainder_batch=False, shuffle=False):
        indices = np.arange(len(self.valid))

        if shuffle:
            np.random.shuffle(indices)

        if use_remainder_batch:
            for i in range(0, len(self.valid), batchsize):
                yield self.valid[indices[i:i+batchsize]]
        else:
            for i in range(0, len(self.valid) - batchsize + 1, batchsize):
                yield self.valid[indices[i:i+batchsize]]

    def get_test_batch(self, batchsize, use_remainder_batch=False, shuffle=True):
        indices = np.arange(len(self.test))

        if shuffle:
            np.random.shuffle(indices)

        if use_remainder_batch:
            for i in range(0, len(self.test), batchsize):
                yield self.test[indices[i:i+batchsize]]
        else:
            for i in range(0, len(self.test) - batchsize + 1, batchsize):
                yield self.test[indices[i:i+batchsize]]

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
    def __init__(self):
        IterableDataset.__init__(self)
        self.toks = [ '0', 'S', 'E', 'U']
        self.char2idx = {}
        self.idx2char = []
        self.max_sent_length = 0
        self.line_lengths = {}

        self.load_stats()
        self.load_data()

    def load_stats(self):
        for t in self.toks:
            self.char2idx[t] = len(self.char2idx)
            self.idx2char += [t]

        for fname in files:
            with open(fname, 'r') as f:
                self.line_lengths[fname] = 0
                for l in f:
                    self.line_lengths[fname] += 1
                    l = l.strip().replace('<unk>', 'U')
                    self.max_sent_length = max(len(l), self.max_sent_length)
                    chars = list(l)
                    for c in chars:
                        if c not in self.char2idx:
                            self.char2idx[c] = len(self.char2idx)
                            self.idx2char += [c]

        self.max_sent_length += 2 # since there's always an sos and eos token

    def load_data(self):
        self.train = np.zeros((self.line_lengths[files[1]], self.max_sent_length), dtype='uint16')
        self.valid = np.zeros((self.line_lengths[files[0]], self.max_sent_length), dtype='uint16')
        self.test = np.zeros((self.line_lengths[files[2]], self.max_sent_length), dtype='uint16')

        file_mapping = {}
        file_mapping[files[0]] = self.valid
        file_mapping[files[1]] = self.train
        file_mapping[files[2]] = self.test

        for fname in files:
            file_mapping[fname][:, 0] = self.char2idx['S']
            with open(fname, 'r') as f:
                for i, l in enumerate(f):
                    l = l.strip().replace('<unk>', 'U')
                    chars = list(l)
                    for j, w in enumerate(chars):
                        file_mapping[fname][i, j+1] = self.char2idx[w]
                    file_mapping[fname][i, len(chars)+1] = self.char2idx['E']