# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:04:58 2020

@author: Oliver
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.layers import MaxPooling3D, Conv3D, Dropout
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


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

    def perform_n_fold_validation(self, X, y, no_epoch, n):
        # n-fold cross validation
        num_val_samples = len(X) / n
        all_acc = []
        all_acc_history = []
        all_loss = []
        all_loss_history = []

        for i in range(n):
            print("Fold:%2i" % i)

            start_partition_index = i * num_val_samples
            end_partition_index = (i + 1) * num_val_samples
            val_X = X[start_partition_index:end_partition_index]
            val_y = X[start_partition_index:end_partition_index]

            train_X = np.concatenate([X[:start_partition_index], X[end_partition_index:]], axis=0)
            train_y = np.concatenate([y[:start_partition_index], y[end_partition_index:]], axis=0)

            history = self.cnn_model.fit(train_X, train_y,
                                         validation_data=(val_X, val_y), batch_size=16, epochs=no_epoch)

            loss_history = history.history['val_loss']
            accuracy_history = history.history['val_acc']

            all_acc_history.append(accuracy_history)
            all_loss_history.append(loss_history)

            val_loss, val_acc = self.cnn_model.evaluate(val_X, val_y, verbose=0)
            all_acc.append(val_acc)
            all_loss.append(val_loss)

    def plot_model_activations(self, X_test):
        # Extracts the outputs of the top 4 layers
        layer_outputs = [layer.output for layer in self.cnn_model.layers[:4]]
        layer_names = [layer.name for layer in self.cnn_model.layers[:4]]

        # Creates a model that will return these outputs, given the model input
        activation_model = models.Model(inputs=self.cnn_model.input, outputs=layer_outputs)
        activation_model.trainable = False

        no_of_tests = 1
        scale = no_of_tests * 5
        fig1, ax1 = plt.subplots(no_of_tests, 3, figsize=[10, scale])

        # Loop adds to plots
        k = 0

        for index in range(0, no_of_tests):
            # image predictions
            test_image = np.expand_dims(X_test[index], axis=0)

            # Returns a list of five Numpy arrays: one array per layer activation
            activations = activation_model.predict(test_image)

            # get activation matrices of layers
            first_layer_activation = activations[1]
            first_layer_n_features = first_layer_activation.shape[-1]
            second_layer_activation = activations[3]
            second_layer_n_features = second_layer_activation.shape[-1]

            # Sum contributions of activations after first Layer
            first_conv_total = np.zeros(first_layer_activation[0, :, :, 0].shape)
            for i in range(0, first_layer_n_features):
                first_conv_total += first_layer_activation[0, :, :, i]

            # Sum contributions of activations after second Layer
            second_conv_total = np.zeros(second_layer_activation[0, :, :, 0].shape)
            for i in range(0, second_layer_n_features):
                second_conv_total += second_layer_activation[0, :, :, i]

            # Plot graph
            ax1[k, 0].imshow(X_test[index])
            ax1[k, 0].axis('off')
            ax1[k, 1].imshow(first_conv_total, cmap='gray')
            ax1[k, 1].axis('off')
            ax1[k, 2].imshow(second_conv_total, cmap='gray')
            ax1[k, 2].axis('off')
            k += 1

        ax1[0, 0].set_title("CT Image")
        ax1[0, 1].set_title("First Layer")
        ax1[0, 2].set_title("Contour")
        plt.show()
