from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np


def get_input(filename):
    img = np.load(filename)
    img = np.reshape(img, (512, 512, -1, 3))
    img = np.divide(img, 255)
    return img


def get_output(filename):
    labels = np.genfromtxt(filename, delimiter=',')
    return np.int_(labels)


def get_file_path(filename):
    slashcount = 0
    output_path = ""
    for i, c in enumerate(reversed(filename)):
        if c == '\\' or c == '/':
            slashcount += 1

        if slashcount == 0:
            output_path = filename[-(i + 1):-4]

    return output_path


def get_training_data(feature_files, label_files):
    batch_X = []
    batch_y = []

    for i, feature in enumerate(feature_files):
        if get_file_path(label_files[i]) != get_file_path(feature):
            print(get_file_path(label_files[i]))
            print(get_file_path(feature))

    for file in feature_files:
        batch_X.append(get_input(file))
    for file in label_files:
        batch_y.append(get_output(file))
    batch_X = np.array(batch_X)
    batch_y = np.array(batch_y)

    return np.array(batch_X), batch_y


class ImageLoaderModel(Sequence):

    def __init__(self, feature_files, label_files, batch_size):
        self.files = [file for file in zip(feature_files, label_files)]
        self.x, self.y = feature_files, label_files
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_index1 = idx * self.batch_size
        batch_index2 = (idx + 1) * self.batch_size

        batch_paths_X = self.x[batch_index1:batch_index2]
        batch_paths_y = self.y[batch_index1:batch_index2]
        batch_X, batch_y = get_training_data(batch_paths_X, batch_paths_y)

        return batch_X, batch_y

    def on_epoch_end(self):
        """Shuffles after each epoch"""
        np.random.shuffle(self.files)
        print("Shuffle")
        self.x = []
        self.y = []

        for features, label in self.files:
            self.x.append(features)
            self.y.append(label)


def get_data_for_model_prediction(feature_files, index):
    print(feature_files[index])
    img = np.load(feature_files[index])
    return np.expand_dims(img, axis=0)
