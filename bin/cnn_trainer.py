from training_data_model import TrainingDataModel
from image_loader_model import ImageLoaderModel
import numpy as np
from tensorflow.keras import models
import matplotlib.pyplot as plt


def create_metrics_plots(epoch_nums, path, metric1, metric2, xlabel, ylabel, legend, loc, yscale=None):
    fig, ax = plt.subplots(1)
    ax.plot(epoch_nums, metric1, 'x')
    ax.plot(epoch_nums, metric2, 'x')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xmin=0)
    ax.legend(legend, loc=loc)
    if yscale is not None:
        ax.set_yscale(yscale)
    fig.savefig(path)
    plt.show(fig)


def save_metrics(history, path):
    np.savetxt(path + "/Validation Accuracy.csv", np.array(history.history['val_accuracy']), delimiter=",")
    np.savetxt(path + "/Validation Loss.csv", np.array(history.history['val_loss']), delimiter=",")
    np.savetxt(path + "/Accuracy.csv", np.array(history.history['accuracy']), delimiter=",")
    np.savetxt(path + "/Loss.csv", np.array(history.history['loss']), delimiter=",")

    epoch_nums = []
    for i in range(len(history.history['accuracy'])):
        epoch_nums.append(i + 1)

        diff = [1 - acc for acc in history.history['accuracy']]
        val_diff = [1 - val_acc for val_acc in history.history['val_accuracy']]

        create_metrics_plots(epoch_nums, path + "/Accuracy.png", history.history['accuracy'],
                             history.history['val_accuracy'],
                             'Accuracy', 'Epoch', ['Training', 'Validation'], 'lower right')

        create_metrics_plots(epoch_nums, path + "/Error.png", diff, val_diff,
                             'Error', 'Epoch', ['Training', 'Validation'], 'upper right', "log")

        create_metrics_plots(epoch_nums, path + "/Loss.png", history.history['loss'], history.history['val_loss'],
                             'Loss', 'Epoch', ['Training', 'Validation'], 'upper right', "log")


class CnnTrainer:

    def __init__(self, allowed_organs, training_data, testing_data, index1, index2, cnn_model):
        X_data = np.concatenate([training_data.features[:index1], training_data.features[index2:]], axis=0)
        y_data = np.concatenate([training_data.labels[:index1], training_data.labels[index2:]], axis=0)

        training_data = TrainingDataModel(training_data.features[index1:index2], training_data.labels[index1:index2])
        validation_data = TrainingDataModel(X_data, y_data)

        self.no_classes = len(allowed_organs)
        self.testing_data = testing_data
        self.stepSize = 2
        self.epoch_no = 8

        self.X_train = training_data.features
        self.y_train = training_data.labels
        self.X_val = validation_data.features
        self.y_val = validation_data.labels
        self.X_test = testing_data.features
        self.y_test = testing_data.labels

        self.training_generator = ImageLoaderModel(self.X_train, self.y_train, self.stepSize)
        self.validation_generator = ImageLoaderModel(self.X_val, self.y_val, self.stepSize)

        temp_batch, _ = self.training_generator.__getitem__(0)
        self.training_data_shape = temp_batch.shape[1:]
        self.training_steps_per_epoch = int(round(len(self.X_train)) / self.stepSize)
        self.validation_steps_per_epoch = int(round(len(self.X_val)) / self.stepSize)
        self.testing_steps = int(round(len(self.X_test)) / self.stepSize)

        self.convolutional_neural_network = cnn_model


    def train_neural_network(self):
        self.cnn_model = self.convolutional_neural_network.compile_and_get_model(no_classes=self.no_classes)

        history = self.cnn_model.fit(self.training_generator,
                                     steps_per_epoch=self.training_steps_per_epoch,
                                     validation_data=self.validation_generator,
                                     validation_steps=self.validation_steps_per_epoch,
                                     epochs=self.epoch_no, verbose=2, max_queue_size=1)
        save_metrics(history, "/target")

        self.history = history

    def get_metrics(self):
        return self.history.history['accuracy'], self.history.history['loss'], \
               self.history.history['val_loss'], self.history.history['val_acc']

    def evaluate_neural_network(self):
        testing_generator = ImageLoaderModel(self.X_test, self.y_test, self.stepSize)
        test_loss, test_accuracy = self.cnn_model.evaluate(testing_data=testing_generator,
                                                           testing_steps=self.testing_steps, verbose=0)
        return test_accuracy, test_loss

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
