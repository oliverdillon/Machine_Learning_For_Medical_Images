import unittest

from models.ctimage import CTImage


class TestCTImage(unittest.TestCase):
    def setUp(self):
        self.test_happy_read = 'C:\\Users\\Oliver\\Documents\\ME\\Coding ' \
                               'Projects\\GitHub\\Machine-Learning-For-Medical-Images\\./HNSCC/HNSCC-01-0001/12-01' \
                               '-1998-NA-PETCT HEAD  NECK CA-92442/2.000000-CT Atten Cor Head IN-25068\\1-001.dcm '
        self.test_unhappy_read = ''

    def test_happy_scenario (self):
        ctimage = CTImage(self.test_happy_read)
        self.assertTrue(hasattr(ctimage, "pydicom"))
        self.assertTrue(hasattr(ctimage, "dicomparser"))
        self.assertTrue(hasattr(ctimage, "zlocation"))
        self.assertTrue(isinstance(ctimage.zlocation, float))

    def does_not_have_z_location(self):
        #TODO find files without z location
        pass
        ctimage = CTImage(self.test_unhappy_read)
        self.assertTrue(hasattr(ctimage, "pydicom"))
        self.assertTrue(hasattr(ctimage, "dicomparser"))
        self.assertTrue(hasattr(ctimage, "zlocation"))
        self.assertTrue(ctimage.zlocation is None)
