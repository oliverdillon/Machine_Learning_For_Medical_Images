from models.training_image_loader import Training_image_loader
from processors.create_cnn_models import Convolutional_neural_network
import numpy as np
from tensorflow.keras import models
import matplotlib.pyplot as plt

def save_metrics(history,path):
    np.savetxt(path+"/Validation Accuracy.csv", np.array(history.history['val_accuracy']), delimiter=",")
    np.savetxt(path+"/Validation Loss.csv", np.array(history.history['val_loss']), delimiter=",")
    np.savetxt(path+"/Accuracy.csv", np.array(history.history['accuracy']), delimiter=",")
    np.savetxt(path+"/Loss.csv", np.array(history.history['loss']), delimiter=",")


    epochNums =[]
    for i in range(len(history.history['accuracy'])):
        epochNums.append(i+1)
        # Plot training & validation accuracy values
        fig, ax = plt.subplots(1)
        ax.plot(epochNums,history.history['accuracy'],'x')
        ax.plot(epochNums,history.history['val_accuracy'],'x')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_xlim(xmin=0)
        ax.legend(['Training', 'Validation'], loc='lower right')
        fig.savefig(path+"/Accuracy.png")
        plt.show(fig)

        diff = [1-acc for acc in history.history['accuracy'] ]
        val_diff = [1-val_acc for val_acc in history.history['val_accuracy'] ]

        # Plot training & validation accuracy values
        fig, ax = plt.subplots(1)
        ax.plot(epochNums,diff,'x')
        ax.plot(epochNums,val_diff,'x')
        ax.set_ylabel('Error')
        ax.set_yscale("log")
        ax.set_xlabel('Epoch')
        ax.set_xlim(xmin=0)
        ax.legend(['Training', 'Validation'], loc='upper right')
        fig.savefig(path+"/Error.png")
        plt.show(fig)


        # Plot training & validation loss values
        fig, ax = plt.subplots(1)
        ax.plot(epochNums,history.history['loss'],'x')
        ax.plot(epochNums,history.history['val_loss'],'x')
        ax.set_ylabel('Loss')
        ax.set_yscale("log")
        ax.set_xlabel('Epoch')
        ax.set_xlim(xmin=0)
        ax.legend(['Training', 'Validation'], loc='upper right')
        fig.savefig(path+"/Loss.png")
        plt.show(fig)



class Train_neural_network:
    def __init__(self,training_data,testing_data,index1,index2):
        self.allowed_organs = ["Right Parotid","Left Parotid"]
        self.no_classes = len(self.allowed_organs)
        self.training_data = training_data
        self.testing_data = testing_data
        self.stepSize = 2
        self.epoch_no = 15

        self.training_steps_per_epoch = int(round(len(self.X_train))/self.stepSize)
        self.validation_steps_per_epoch = int(round(len(self.X_Val))/self.stepSize)

        self.X_train = self.training_data.dataset[index2:]
        self.y_train = self.training_data.labels[index2:]
        self.X_Val = self.training_data.dataset[index1:index2]
        self.y_Val = self.training_data.labels[index1:index2]

        self.convolutional_neural_network = Convolutional_neural_network(no_of_spacial_dimensions=3)

    def train_neural_network(self):
        training_generator = Training_image_loader(self.X_train, self.y_train, self.stepSize)
        validation_generator = Training_image_loader(self.X_Val, self.y_Val, self.stepSize)

        self.convolutional_neural_network.add_convolution_layer(no_filter=32, shape=self.training_data.dataset[0].shape)
        self.convolutional_neural_network.add_max_pooling_layer()
        self.convolutional_neural_network.add_convolution_layer(no_filter=64)
        self.convolutional_neural_network.add_max_pooling_layer()
        self.convolutional_neural_network.add_dense_layer(no_filter=512)
        self.cnn_model = self.convolutional_neural_network.compile_and_get_model(no_classes=2)

        history = self.cnn_model.fit_generator(generator=training_generator, steps_per_epoch=self.training_steps_per_epoch,
                                      validation_data=validation_generator,
                                      validation_steps=self.validation_steps_per_epoch,
                                      epochs=self.epoch_no, verbose=2, max_queue_size=1)
        save_metrics(history, "/target")

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
