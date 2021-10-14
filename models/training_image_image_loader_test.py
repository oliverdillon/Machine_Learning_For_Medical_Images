import unittest
import numpy as np
import os

from models.training_image_loader import Training_image_loader


class TestCTImage(unittest.TestCase):
    def setUp(self):
        self.batch_size =2
        self.feature_files = [
            "/../target/HNSCC-01-0001/Left_Parotid_feature.npy"
        ]
        self.label_files = [
            "/../target/HNSCC-01-0001/Left_Parotid_label.txt"
        ]
        for i, label in enumerate(self.label_files):
            self.label_files[i] = os.getcwd()+label
            self.feature_files[i] = os.getcwd() + self.feature_files[i]


    def test_happy_scenario (self):
        imageloader = Training_image_loader(self.feature_files,self.label_files,self.batch_size)
        actual_feature_file_batch , actual_label_file_batch = imageloader.__getitem__(0)
        expected_feature_file_batch = []
        expected_label_file_batch = []
        for i, label in enumerate(self.label_files):
            expected_feature_file = np.load(self.feature_files[i])
            expected_feature_file = np.divide(expected_feature_file,255)
            expected_feature_file_batch.append(expected_feature_file)

            expected_label_file = np.genfromtxt(label, delimiter = ',')
            expected_label_file_batch.append(expected_label_file)


        self.assertAlmostEqual(imageloader.batch_size,2)
        np.testing.assert_almost_equal(expected_feature_file_batch,actual_feature_file_batch)
        np.testing.assert_almost_equal(expected_label_file_batch,actual_label_file_batch)

    def test_unhappy_file_does_not_exist (self):
        imageloader = Training_image_loader(self.feature_files,self.label_files,self.batch_size)

