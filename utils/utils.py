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
sys.path.append(os.path.abspath('../data_utils'))
from data_utils import data_utils

class vgg_block(layers.Layer):
    def __init__(self, n_filters: int):
        """
        :param n_filters: number of filters of the output image
        """
        super().__init__()
        self.n_filters = n_filters
        self.conv1 = layers.Conv2D(n_filters, (3,3), padding='same', activation='relu')
        self.BN1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(n_filters, (3,3), padding='same', activation='relu')
        self.BN2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(n_filters, (3,3), padding='same', activation='relu')
        self.BN3 = layers.BatchNormalization()
    def call (self, layer_in: tf.Tensor, training: bool=False):
        """
        :param layer_in: the input image
        :return layer_in: the image created by applying a vgg block on the input image
        """
        layer_in = self.conv1(layer_in)
        layer_in = self.BN1(layer_in)
        layer_in = self.conv2(layer_in)
        layer_in = self.BN2(layer_in)
        layer_in = self.conv3(layer_in)
        layer_in = self.BN3(layer_in)
        layer_in = layers.AveragePooling2D((2,2), strides=(2,2))(layer_in)
        return layer_in
    pass

class my_model(tfk.Model):
    def __init__(self):
        """
        instantiating a CNN + LSTM model
        """
        super().__init__()
        image_height, image_width = 50, 200
        n_timesteps_in = 5
        n_features = 19
        self.augm1 = tf.keras.layers.RandomBrightness(0.4)
        self.augm2 = tf.keras.layers.RandomContrast(0.4)
        self.augm3 = tf.keras.layers.RandomRotation(0.01, fill_mode='nearest', interpolation='bilinear')
        self.augm4 = tf.keras.layers.RandomZoom(height_factor=(-0.10, 0.10), fill_mode='nearest')
        self.vgg1 = vgg_block(32)
        self.drop1 = layers.Dropout(0.2)
        self.vgg2 = vgg_block(64)
        self.drop2 = layers.Dropout(0.2)
        # LSTM input: A 3D tensor with shape [batch, timesteps, feature].
        # we want the "timesteps" dimension of LSTM input to be something meaningful.
        # as humans read from left-to-right or right-to-left, we want model to have the same approach
        # so we treat "width" dimension as timesteps, so we use reshape like below:
        self.reshape1 = layers.Reshape(((int(image_width / 4), -1)))
        self.lstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))
        self.reshape2 = layers.Reshape(((n_timesteps_in, -1)))
        self.dense1 = layers.Dense(256)
        self.dense2 = layers.Dense(n_features, activation='softmax')
    def call(self, inputs: tf.Tensor, training=False):
        """
        :param inputs: the input image
        :return: output of shape (batchsize, n_timesteps_in, n_features)
        """
        x = self.augm1(inputs)
        x = self.augm2(x)
        x = self.augm3(x)
        x = self.augm4(x)
        x = self.vgg1(x)
        x = self.drop1(x)
        x = self.vgg2(x)
        x = self.drop2(x)
        x = self.reshape1(x)
        x = self.lstm(x)
        x = self.reshape2(x)
        x = self.dense1(x)
        return self.dense2(x)
    def model(self):
        """
        :return: a model created by using keras functional API
        """
        image_height, image_width = 50, 200
        x = layers.Input(shape=(image_height, image_width, 1))
        return Model(inputs=x, outputs=self.call(x))

def start_training(model: tfk.Model, ds_train: tf.data.Dataset, ds_test: tf.data.Dataset, enc: sklearn.preprocessing._encoders.OneHotEncoder, num_epochs:int, batch_size:int=64):
    """
    :param model: the model we are going to train
    :param ds_train: the data generator we use for training
    :param enc: a fitted sklearn OneHotEncoder to transform string labels to matrix labels
    :param num_epochs: the number of training epochs
    :param batch_size: the batch size 
    :return lossess: a list containing loss for each epoch
    :return accuracies: a list containing accuracies on train-set for each epoch
    :return test_accuracies: a list containing accuracies on test-set for each epoch
    """
    n_timesteps_in = 5
    n_features = 19
    lossess, accuracies, test_accuracies = [], [], []
    criterion = tfk.losses.CategoricalCrossentropy()
    optimizer = tfk.optimizers.Adam()
    acc_metric = tfk.metrics.CategoricalAccuracy()
    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(ds_train.map(data_utils.process_path).batch(batch_size)):
            y = np.array(y).astype('<U7')
            y = data_utils.label_to_array(y, enc)
            y = y.reshape(-1, n_timesteps_in, n_features)
            x = data_utils.process_image(x, training=True)
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = criterion(y, y_pred)
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            acc_metric.update_state(y, y_pred)
            print(f"Batch {batch_idx + 1} completed")
        train_acc = acc_metric.result()
        lossess.append(loss)
        accuracies.append(train_acc)
        print(f"accuracy on train-set after epoch {epoch+1} is {train_acc}")
        acc_metric.reset_state()
        test_acc = start_testing(model, ds_test, enc)
        test_accuracies.append(test_acc)

    return lossess, accuracies, test_accuracies

def start_testing(model: tfk.Model, ds_test: tf.data.Dataset, enc: sklearn.preprocessing._encoders.OneHotEncoder):
    """
    :param model: the model we are going to test
    :param ds_test: the data generator we use for testing
    :param enc: a fitted sklearn OneHotEncoder to transform string labels to matrix labels
    :return test_acc: accuracy on test set
    """
    n_timesteps_in = 5
    n_features = 19
    acc_metric = tfk.metrics.CategoricalAccuracy()
    acc_metric.reset_state()
    for batch_idx, (x, y) in enumerate(ds_test.map(data_utils.process_path).batch(64)):
        y = np.array(y).astype('<U7')
        y = data_utils.label_to_array(y, enc)
        y = y.reshape(-1, n_timesteps_in, n_features)
        x = data_utils.process_image(x, training=True)
        y_pred_test = model(x)
        acc_metric.update_state(y, y_pred_test)
    test_acc = acc_metric.result()
    print(f"accuracy on test set is {test_acc}")
    return test_acc
    

