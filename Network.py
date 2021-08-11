# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:43:05 2020

@author: Dim
"""

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Concatenate, BatchNormalization
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers.core import Activation
import numpy as np


def ConvBnRelu(x, kernel, channel, BN):
    if BN == True:
        x = layers.Conv2D(channel, kernel, strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        return x
    else:
        x = layers.Conv2D(channel, kernel, strides=(1, 1), padding='same')(x)
        x = Activation('relu')(x)
        return x


def makeModel(inputShape, kernel, featuremap, learningrate, skip_connection, transposed, BN):
    reverseF = np.fliplr([featuremap])
    inp = Input(shape=(inputShape))

    if skip_connection == True:  # skip connection
        layers_skip = list()
        for i in range(len(featuremap)):  # Downsampling'
            if i == 0:
                x = ConvBnRelu(inp, kernel, featuremap[i], BN)
                layers_skip.append(x)
                x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
            else:
                x = ConvBnRelu(x, kernel, featuremap[i], BN)
                layers_skip.append(x)
                x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        for j in range(reverseF.shape[-1]):  # Upsampling
            if transposed == False:
                x = layers.UpSampling2D(input_shape=x.shape)(x)
                x = ConvBnRelu(x, kernel, reverseF[0][j], BN)
                x = Concatenate(
                    axis=-1)([x, layers_skip[(len(layers_skip)-1)-j]])
            else:
                if BN == True:
                    x = layers.Conv2DTranspose(
                        reverseF[0][j], (2, 2), strides=(2, 2), padding='valid')(x)
                    x = BatchNormalization(axis=-1)(x)
                    x = Activation('relu')(x)
                    x = Concatenate(
                        axis=-1)([x, layers_skip[(len(layers_skip)-1)-j]])
                else:
                    x = layers.Conv2DTranspose(
                        reverseF[0][j], (2, 2), strides=(2, 2), padding='valid')(x)
                    x = Concatenate(
                        axis=-1)([x, layers_skip[(len(layers_skip)-1)-j]])
    else:  # skip connection
        for i in range(len(featuremap)):  # Downsampling
            if i == 0:
                x = ConvBnRelu(inp, kernel, featuremap[i], BN)
                x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
            else:
                x = ConvBnRelu(x, kernel, featuremap[i], BN)
                x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        for j in range(reverseF.shape[-1]):  # Upsampling
            if transposed == False:
                x = layers.UpSampling2D(input_shape=x.shape)(x)
                x = ConvBnRelu(x, kernel, reverseF[0][j], BN)
            else:
                if BN == True:
                    x = layers.Conv2DTranspose(
                        reverseF[0][j], (2, 2), strides=(2, 2), padding='valid')(x)
                    x = BatchNormalization(axis=-1)(x)
                    x = Activation('relu')(x)
                else:
                    x = layers.Conv2DTranspose(
                        reverseF[0][j], (2, 2), strides=(2, 2), padding='valid')(x)

    out = layers.Conv2D(5, (1, 1), strides=(1, 1), padding='same')(x)
    out = Activation('sigmoid')(out)

    model = Model(inp, out)
    model.summary()
    optimizers = tf.keras.optimizers.Adam(lr=learningrate)  # KURANG BAGUS
    # optimizers=tf.keras.optimizers.SGD(lr=learningrate)

    model.compile(loss='mse', optimizer=optimizers)
    return model
