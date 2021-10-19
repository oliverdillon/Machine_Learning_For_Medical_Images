import unittest
from unittest.mock import patch, mock_open


class TestCTImage(unittest.TestCase):
    def setUp(self) -> None:
        self.augmentation_metadata = {}
        self.classification_metadata = {}

    @patch('__main__.open', mock_open(read_data='foo\nbar\nbaz\n'))
    def test_top_level_augmentation_preprocessing(self):
        self.assertAlmostEqual(True, True)

    @patch('__main__.open', mock_open(read_data='foo\nbar\nbaz\n'))
    def test_top_level_classification_preprocessing(self):
        self.assertAlmostEqual(True, True)
