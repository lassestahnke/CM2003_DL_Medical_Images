import nibabel as nib
from tensorflow.keras.utils import Sequence
import numpy as np


def load_streamlines(dataPath, subject_ids, bundles, n_tracts_per_bundle):
    X = []
    y = []
    for i in range(len(subject_ids)):
        for c in range((len(bundles))):
            filename = dataPath + subject_ids[i] + '/' + bundles[c] + '.trk'
            tfile = nib.streamlines.load(filename)
            streamlines = tfile.streamlines
            n_tracts_total = len(streamlines)
            ix_tracts = np.random.choice(range(n_tracts_total),
                                         n_tracts_per_bundle,
                                         replace=False)

            streamlines_data = streamlines.data
            streamlines_offsets = streamlines._offsets

            for j in range(n_tracts_per_bundle):
                ix_j = ix_tracts[j]
                offset_start = streamlines_offsets[ix_j]
                if ix_j < (n_tracts_total - 1):
                    offset_end = streamlines_offsets[ix_j + 1]
                    streamline_j = streamlines_data[offset_start:offset_end]
                else:
                    streamline_j = streamlines_data[offset_start:]
                X.append(np.asarray(streamline_j))
                y.append(c)
    return X, y


class MyBatchGenerator(Sequence):
    def __init__(self, X, y, batch_size=1, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """ Get number of batches per epoch """
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        """ Shuffle indexes after each epoch """

        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *self.X[index].shape))
        yb = np.empty((self.batch_size, 1))
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
            yb[s] = self.y[index]
        return Xb, yb
