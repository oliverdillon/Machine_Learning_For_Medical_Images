from models.cnn_dataset import CNN_Dataset
from processors.train_neural_networks import Train_neural_network
from util.file_functions import read_txt_and_append_to_list


class Perform_n_fold_validation:

    def __init__(self, allowed_organs, no_of_folds, directory):
        self.allowed_organs = allowed_organs
        self.no_of_folds = no_of_folds
        self.base_directory = directory
        self.get_data_directories()
        self.perform_n_fold_validation()


    def get_data_directories(self):
        total_features = []
        total_labels = []
        read_txt_and_append_to_list(self.base_directory + "/features.txt", total_features)
        read_txt_and_append_to_list(self.base_directory  + "/labels.txt", total_labels)
        self.training_size = int(round(len(total_features)*0.6))
        self.training_dataset = CNN_Dataset(total_features[:self.training_size],total_labels[:self.training_size])
        self.testing_dataset = CNN_Dataset(total_features[self.training_size:],total_labels[self.training_size:])
        print('read directory files')

    def perform_n_fold_validation(self):
        # n-fold cross validation
        num_val_samples = int(self.training_size / self.no_of_folds)
        accuracy_values = []
        loss_values = []
        validation_accuracy_values = []
        validation_loss_values = []
        testing_accuracy_values = []
        testing_loss_values = []

        for i in range(self.no_of_folds):
            print("Fold:%2i" % i)

            start_partition_index = i * num_val_samples
            end_partition_index = (i + 1) * num_val_samples
            train_neural_network = Train_neural_network(self.allowed_organs, self.training_dataset, self.testing_dataset,
                                                        start_partition_index, end_partition_index)
            accuracy, loss, validation_accuracy, validation_loss = train_neural_network.get_metrics()
            testing_accuracy, testing_loss = train_neural_network.evaluate_neural_network()
            accuracy_values.append(accuracy)
            loss_values.append(loss)

            validation_accuracy_values.append(validation_accuracy)
            validation_loss_values.append(validation_loss)

            testing_accuracy_values.append(testing_accuracy)
            testing_loss_values.append(testing_loss)
