from processors.train_neural_networks import Train_neural_network
class Perform_n_fold_validation:

    def __init__(self, allowed_organs, no_of_folds):
        self.allowed_organs = allowed_organs
        self.no_of_folds = no_of_folds
        self.training_size = 7
        self.perform_n_fold_validation()


    def get_data_directories(self):
        ##ra
        print('read directory files')

    def perform_n_fold_validation(self):
        # n-fold cross validation
        num_val_samples = self.training_size / self.no_of_folds
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
            train_neural_network = Train_neural_network(self.allowed_organs, None, None,
                                                        start_partition_index, end_partition_index)
            accuracy, loss, validation_accuracy, validation_loss = train_neural_network.get_metrics()
            testing_accuracy, testing_loss = train_neural_network.evaluate_neural_network()
            accuracy_values.append(accuracy)
            loss_values.append(loss)

            validation_accuracy_values.append(validation_accuracy)
            validation_loss_values.append(validation_loss)

            testing_accuracy_values.append(testing_accuracy)
            testing_loss_values.append(testing_loss)
