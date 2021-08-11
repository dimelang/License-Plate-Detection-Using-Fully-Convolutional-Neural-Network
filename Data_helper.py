import numpy as np
import h5py
import pickle


def loadDataset(tipe, *argv):
    if tipe == "train":
        if argv[0] == False:
            X = h5py.File('Dataset/X_train.hdf5', 'r')
            X = np.array(X["default"][:])
            X = np.expand_dims(X, axis=-1)

            y = h5py.File('Dataset/y_train.hdf5', 'r')
            y = np.array(y["default"][:])

            with open('Dataset/y_trainLabel.pkl', 'rb') as f:
                y_label = pickle.load(f)

            y_label = y_label[:]
            print("load data", tipe, "sukses")
            print(np.array(X).shape, np.array(
                y).shape, len(y_label), "Data ", tipe)
        else:
            X = h5py.File('Dataset/X_train.hdf5', 'r')
            X = np.array(X["default"][argv[1]])
            X = np.expand_dims(X, axis=-1)

            y = h5py.File('Dataset/y_train.hdf5', 'r')
            y = np.array(y["default"][argv[1]])

            with open('Dataset/y_trainLabel.pkl', 'rb') as f:
                y_label = pickle.load(f)

            y_label = [y_label[i] for i in argv[1]]

        return X, y, y_label
    elif tipe == "test":
        X = h5py.File('Dataset/X_test.hdf5', 'r')
        X = np.array(X["default"][:])
        X = np.expand_dims(X, axis=-1)

        y = h5py.File('Dataset/y_test.hdf5', 'r')
        y = np.array(y["default"][:])

        with open('Dataset/y_testLabel.pkl', 'rb') as f:
            y_label = pickle.load(f)

        y_label = y_label[:]

        return X, y, y_label
