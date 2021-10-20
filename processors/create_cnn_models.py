# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:04:58 2020

@author: Oliver
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.layers import MaxPooling3D, Conv3D, Dropout
from tensorflow.keras.optimizers import Adam



class Convolutional_neural_network:
    def __init__(self, no_of_spacial_dimensions):
        self.no_of_spacial_dimensions = no_of_spacial_dimensions
        self.cnn_model = Sequential()  # add each layer in turn

    def add_convolution_layer(self, strides=(3, 3, 3), no_filter=32, shape=None):
        if self.no_of_spacial_dimensions == 3:
            self.cnn_model.add(Conv3D(no_filter, strides, input_shape=shape, activation='relu'))
        elif self.no_of_spacial_dimensions == 2:
            self.cnn_model.add(Conv2D(no_filter, (3, 3), input_shape=shape, activation='relu'))

    def add_max_pooling_layer(self, strides=(2, 2, 2)):
        if self.no_of_spacial_dimensions == 3:
            self.cnn_model.add(MaxPooling3D(pool_size=strides))
        elif self.no_of_spacial_dimensions == 2:
            self.cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    def add_dense_layer(self, no_filter=128):
        self.cnn_model.add(Flatten())
        self.cnn_model.add(Dense(units=no_filter, activation='relu'))

    def add_dropout_layer(self, no_filter=128):
        self.cnn_model.add(Dropout(no_filter))

    def compile_and_get_model(self, no_classes, lr=0.001):
        if no_classes == 2:
            activ = 'sigmoid'
            loss_func = 'binary_crossentropy'
        else:
            activ = 'softmax'
            loss_func = 'categorical_crossentropy'

        # Output Layer
        self.cnn_model.add(Dense(units=no_classes, activation=activ))

        optimize = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

        # Compiling the CNN
        self.cnn_model.compile(optimizer=optimize, loss=loss_func, metrics=['accuracy'])

        print(self.cnn_model.summary())
        return self.cnn_model