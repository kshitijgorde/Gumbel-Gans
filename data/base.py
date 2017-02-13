import numpy as np

class IterableDataset(object):
    def __init__(self):
        self.vocab_size = 0
        self.train = np.zeros((0, 0))
        self.valid = np.zeros((0, 0))
        self.test = np.zeros((0, 0))

    def one_hot(self, idx):
        oh = np.zeros(list(idx.shape) + [self.vocab_size])
        oh[list(np.indices(oh.shape[:-1])) + [idx]] = 1
        return oh

    def get_train_batch(self, batchsize, use_remainder_batch=False, shuffle=True, one_hot=True, max_size=float('inf')):
        indices = np.arange(min(len(self.train), max_size))

        if shuffle:
            np.random.shuffle(indices)

        if use_remainder_batch:
            for i in range(0, len(self.train), batchsize):
                if one_hot:
                    yield self.one_hot(self.train[indices[i:i+batchsize]])
                else:
                    yield self.train[indices[i:i+batchsize]]
        else:
            for i in range(0, len(self.train) - batchsize + 1, batchsize):
                if one_hot:
                    yield self.one_hot(self.train[indices[i:i+batchsize]])
                else:
                    yield self.train[indices[i:i+batchsize]]

    def get_valid_batch(self, batchsize, use_remainder_batch=False, shuffle=False, one_hot=True):
        indices = np.arange(len(self.valid))

        if shuffle:
            np.random.shuffle(indices)

        if use_remainder_batch:
            for i in range(0, len(self.valid), batchsize):
                if one_hot:
                    yield self.one_hot(self.valid[indices[i:i+batchsize]])
                else:
                    yield self.valid[indices[i:i+batchsize]]
        else:
            for i in range(0, len(self.valid) - batchsize + 1, batchsize):
                if one_hot:
                    yield self.one_hot(self.valid[indices[i:i+batchsize]])
                else:
                    yield self.valid[indices[i:i+batchsize]]

    def get_test_batch(self, batchsize, use_remainder_batch=False, shuffle=False, one_hot=True):
        indices = np.arange(len(self.test))

        if shuffle:
            np.random.shuffle(indices)

        if use_remainder_batch:
            for i in range(0, len(self.test), batchsize):
                if one_hot:
                    yield self.one_hot(self.test[indices[i:i+batchsize]])
                else:
                    yield self.test[indices[i:i+batchsize]]
        else:
            for i in range(0, len(self.test) - batchsize + 1, batchsize):
                if one_hot:
                    yield self.one_hot(self.test[indices[i:i+batchsize]])
                else:
                    yield self.test[indices[i:i+batchsize]]

