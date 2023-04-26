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
from typing import Type

from utils import utils
from data_utils import data_utils

# reducing the number of warning we get, although we still get many warnings ;)
# the layers I used for augmentation are generating those warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#######################################################################################

## pre-processing
directory = "samples/"
ds = data_utils.load_data(directory)
# dividing the dataset to train-set and test-set and creating generators for each
# test/train data ratio
test_portion = 0.2
# splitting the data
ds_train, ds_test = data_utils.split_train_test(ds, test_portion)
# what are the characters in the captcha images
all_charachter = data_utils.extract_all_charachters(ds)
# making an encoder to OneHot-encode string labels 
enc = data_utils.one_hot_encoder(all_charachter)
# the number of supported charachters for model
n_features = len(all_charachter)
# the supported length of captcha labels for model
n_timesteps_in = 5

## defining the training procedure
num_epochs = 3
batch_size = 64
# creating a model
model = utils.my_model().model()
# start training (and testing in each epoch)
lossess, accuracies, test_accuracies = utils.start_training(model, ds_train, ds_test, enc, num_epochs, batch_size)


## saving the result
# saving the plots
data_utils.savefig_loss(lossess)
data_utils.savefig_acc(accuracies, test_accuracies) 
# saving to csv files
data_utils.savecsv_loss(lossess)
data_utils.savecsv_acc(accuracies, "train")
data_utils.savecsv_acc(test_accuracies, "test")
# saving the final accuracies on a .txt file
data_utils.write_report(accuracies[-1], test_accuracies[-1])