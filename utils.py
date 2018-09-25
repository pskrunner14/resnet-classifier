import h5py
import dill
import numpy as np

def load_dataset():
    train_dataset = h5py.File('datasets/train_data.h5', 'r')
    X_train = np.array(train_dataset["X"][:]) # your train set features
    y_train = np.array(train_dataset["y"][:]) # your train set labels

    train_dataset = h5py.File('datasets/val_data.h5', 'r')
    X_val = np.array(train_dataset["X"][:]) # your validation set features
    y_val = np.array(train_dataset["y"][:]) # your validation set labels

    train_dataset = h5py.File('datasets/test_data.h5', 'r')
    X_test = np.array(train_dataset["X"][:]) # your test set features
    y_test = np.array(train_dataset["y"][:]) # your test set labels

    with open('datasets/classes.dill', 'rb') as file:
        categories = dill.load(file)
     
    return X_train, X_val, X_test, y_train, y_val, y_test, categories