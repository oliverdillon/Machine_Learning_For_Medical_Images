import unittest
import numpy as np
import os

from image_loader_model import ImageLoaderModel


def set_absolute_directories(files):
    for i, file in enumerate(files):
        files[i] = os.getcwd() + file


class TestImageLoaderModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.feature_files = [
            "/../test_data/classification/HNSCC-01-0001/Left_Parotid_feature.npy"
        ]
        self.label_files = [
            "/../test_data/classification/HNSCC-01-0001/Left_Parotid_label.txt"
        ]
        set_absolute_directories(self.feature_files)
        set_absolute_directories(self.label_files)

    def test_happy_scenario(self):
        image_loader = ImageLoaderModel(self.feature_files, self.label_files, self.batch_size)
        actual_feature_file_batch, actual_label_file_batch = image_loader.__getitem__(0)  # TODO could be replaced
        # with next operator?
        expected_feature_file_batch = []
        expected_label_file_batch = []
        for i, label in enumerate(self.label_files):
            expected_feature_file = np.load(self.feature_files[i])
            expected_feature_file = np.divide(expected_feature_file, 255)
            expected_feature_file_batch.append(expected_feature_file)

            expected_label_file = np.genfromtxt(label, delimiter=',')
            expected_label_file_batch.append(expected_label_file)

        self.assertAlmostEqual(image_loader.batch_size, 2)
        np.testing.assert_almost_equal(expected_feature_file_batch, actual_feature_file_batch)
        np.testing.assert_almost_equal(expected_label_file_batch, actual_label_file_batch)

    def test_unhappy_feature_file_does_not_exist(self):
        feature_files = [
            "/../test_data/classification/HNSCC-01-0001/feature.npy"
        ]
        set_absolute_directories(feature_files)
        image_loader = ImageLoaderModel(feature_files, self.label_files, self.batch_size)
        self.assertAlmostEqual(image_loader.batch_size, 2)
        self.assertRaises(FileNotFoundError, image_loader.__getitem__, 0)

    def test_unhappy_label_file_does_not_exist(self):
        label_files = [
            "/../test_data/classification/HNSCC-01-0001/label.txt"
        ]
        set_absolute_directories(label_files)
        imageloader = ImageLoaderModel(self.feature_files, label_files, self.batch_size)
        self.assertAlmostEqual(imageloader.batch_size, 2)
        self.assertRaises(IOError, imageloader.__getitem__, 0)
