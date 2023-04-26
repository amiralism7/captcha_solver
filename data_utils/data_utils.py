import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import pathlib
import os
import sklearn.model_selection
from tensorflow.keras import layers, Sequential, Model
from keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder
import sklearn
import sklearn.preprocessing

import sys
sys.path.append(os.path.abspath('../utils'))
from utils import utils

def load_data(directory: str):
    """
    :param directory: a string containing the path of data images
    :return ds: a tf.data.Dataset generator
    returns a tf.data.Dataset generator to be used for training
    """
    ds = tf.data.Dataset.list_files(str(pathlib.Path(directory+"*.png")), shuffle=True, seed=1)
    return ds

def split_train_test(ds: tf.data.Dataset,  test_portion: float=0.2):
    """
    :param ds: a tf.data.Dataset generator
    :return ds_train: a tf.data.Dataset generator for training
    :return ds_test: a tf.data.Dataset generator for testing
    splits the given dataset (generator) into test_set and train_set (generators)
    """
    ds_test = ds.take(int(len(ds) * test_portion))
    ds_train = ds.skip(int(len(ds) * test_portion))
    return ds_train, ds_test

def extract_label(file_path: tf.Tensor):
    """
    :param file_path: a tf.Tensor object created by iterating over a tf.data.Dataset generator 
    :return label: ground truth label for the image with corresponding file_path
    This function is to be used for extracting ground_truth labels from file paths
    """
    # str(file_path) will have a format like below:
    # "tf.Tensor(b'samples/yxd7m.png', shape=(), dtype=string)"
    # we only want "yxd7m" as a label
    label = str(file_path)[str(file_path).find("samples/") + len("samples/"):str(file_path).find(".png")]
    return label

def extract_all_charachters(ds: tf.data.Dataset):
    """
    :param ds: a tf.data.Dataset generator
    :return all_charachter: all characters that are used to create captcha images
    This function is to be used for getting all possible characters in captcha images.
    """
    all_charachter = []
    for file_path in ds:
        label = extract_label(file_path)
        all_charachter.extend(list(label))
    all_charachter = np.array(all_charachter)
    all_charachter = np.unique(all_charachter)
    return all_charachter

def one_hot_encoder(all_charachter: np.array):
    """
    :param all_charachter: all characters that are used to create captcha images
    :return enc: a fitted sklearn OneHotEncoder to transform string labels to matrix labels
    this function is to be used for OneHot-encoding characters, and it returns a sklearn encoder object.
    """
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(all_charachter.reshape(-1, 1))
    return enc

def label_to_array(inputs: str, enc: sklearn.preprocessing._encoders.OneHotEncoder):
    """
    :param inputs: a string label
    :param enc: a fitted sklearn OneHotEncoder to transform string labels to matrix labels
    :return outputs: corresponding OneHotEncoded label to be used in training
    This function uses the one_hot_encoder to encode labels into a (n_timesteps_in, n_features) vector
    """
    outputs = []
    for label in inputs:
        tmp_list = list(label)
        output = [None] * len(label)
        for i in range(len(label)):
            output[i] = enc.transform(np.array(label[i]).reshape(-1,1)).toarray()
        output = np.array(output)
        outputs.append(output)
        
    return np.array(outputs)

def array_to_label(array: np.array, all_charachter: np.array, enc: sklearn.preprocessing._encoders.OneHotEncoder):
    """
    :param array: a OneHotEncoded label
    :param all_charachter: all characters that are used to create captcha images
    :param enc: a fitted sklearn OneHotEncoder to transform string labels to matrix labels
    :return: corresponding string label
    This function converts a (n_timesteps_in, n_features) vector to an array of length (n_timesteps_in, )
    """
    n_timesteps_in = 5
    n_features = 19
    try:
        return all_charachter[np.array(array).argmax(axis=3)].reshape(-1)
    except:
        y = np.array(array).astype('<U7')
        y = label_to_array(y, enc)
        y = y.reshape(-1, n_timesteps_in, n_features)
        return all_charachter[np.array(y).argmax(axis=2)].reshape(-1)

def process_path(file_path: tf.Tensor):
    """
    :param file_path: a tf.Tensor object created by iterating over a tf.data.Dataset generator 
    :return image: image with the corresponding file_path to be used in training
    :return label: ground truth label for the corresponding file_path
    This function is to be applied (by using .map) to a tf.data.Dataset object in order to extract (x, y) for training
    """
    n_timesteps_in = 5
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image)   
    
    label = tf.strings.substr(file_path, pos=len("samples/"), len=n_timesteps_in)
    return image, label

def process_image(image: tf.Tensor, training: bool=False):
    """
    :param image: an image created by reading a .png file with 4 channels
    :return image: an image with one grayscale channel and a determined size
    this function removes extra channels and also makes sure that the image size is acceptable by model.
    """
    image_height, image_width = 50, 200
    image = image[:, :, :, 0:-1]
    image = tf.image.resize(image, (image_height, image_width))
    image = tf.image.rgb_to_grayscale(image)
#     image = tf.keras.layers.RandomBrightness(0.4)(image, training=True)
#     image = tf.keras.layers.RandomContrast(0.4)(image, training=True)
#     image = tf.keras.layers.RandomRotation(0.01, fill_mode='nearest', interpolation='bilinear')(image, training=True)
#     image = tf.keras.layers.RandomZoom(height_factor=(-0.10, 0.10), fill_mode='nearest')(image, training=True)
    return image


def savefig_acc(accuracies: list, test_accuracies: list):
    """
    :param accuracies: a list containig accuracies on train-set for each epoch
    :param test_accuracies: a list containig accuracies on test-set for each epoch
    This function will save an image showing a plot of accuracies over epochs
    """
    plt.plot(np.array(accuracies) * 100, color="r", label="train accuracy")
    plt.plot(np.array(test_accuracies) * 100, color="g", label="test accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")
    plt.legend()
    plt.savefig('results/accuracy.png')
    plt.close()

    
def savefig_loss(lossess: list):
    """
    :param lossess: a list containig losses for each epoch
    This function will save an image showing a plot of losses over epochs
    """
    plt.plot(np.log10(np.array(lossess)))
    plt.title("Log10 (Loss) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Log (loss)")
    plt.savefig('results/loss.png')
    plt.close()

def savecsv_acc(accuracies: list, which: str):
    """
    :param accuracies: a list containig accuracies for each epoch
    :param str: a string that clarifies that accuracy for which dataset is given
    This function will save csv file containing accuracies over epochs
    """
    if which == "train":
        pd.Series(np.array(accuracies)).to_csv("results/accuracies.csv")
    elif which == "test":
        pd.Series(np.array(accuracies)).to_csv("results/accuracies_test.csv")

    
def savecsv_loss(lossess: list):
    """
    :param lossess: a list containig losses for each epoch
    This function will save csv file containing losses over epochs
    """
    pd.Series(np.array(lossess)).to_csv("results/losses.csv")


def write_report(train_acc: tf.float32, test_acc: tf.float32):
    """
    :param train_acc: final accuracy on train-set
    :param test_acc: final accuracy on test-set
    This function will save a txt file containing final accuracy on train-set and test
    """
    with open('results/report.txt', 'w') as f:
        f.write(f'The final accuracy on train-set is {np.array(train_acc)} \n')
        f.write(f'The final accuracy on test-set is {np.array(test_acc)}')