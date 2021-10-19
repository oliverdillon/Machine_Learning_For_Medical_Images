import unittest
from transformation_functions import Coordinate_transformer
import matplotlib.pyplot as plt

class TestCTImage(unittest.TestCase):
    def setUp(self) :
        self.coordinates = []
        self.xcoordinates = [100,100,200,200]
        self.ycoordinates = [100,200,200,100]

        for i, ycoordinate in enumerate(self.ycoordinates):
            self.coordinates.append((self.xcoordinates[i],ycoordinate))

        self.coordinate_transformer = Coordinate_transformer(self.coordinates)
        before_translation = self.coordinate_transformer.get_contoured_image()
        plt.imshow(before_translation)
        plt.close()
        self.test_get_coordinate()

    def test_get_coordinate(self):
        self.assertAlmostEqual(self.coordinate_transformer.x , self.xcoordinates)
        self.assertAlmostEqual(self.coordinate_transformer.y , self.ycoordinates)
        self.assertAlmostEqual(self.coordinate_transformer.ox , 150)
        self.assertAlmostEqual(self.coordinate_transformer.oy , 150)

    def test_contour_translation(self):
        self.coordinate_transformer.translate_contour()
        transformed_contour = self.coordinate_transformer.get_contoured_image()
        plt.imshow(transformed_contour)
        plt.close()

    def test_translate_random_points(self):
        self.coordinate_transformer.move_random_points(0.1)
        transformed_contour = self.coordinate_transformer.get_contoured_image()
        plt.imshow(transformed_contour)
        plt.close()

    def test_shear_points(self):
        self.coordinate_transformer.shear_points()
        transformed_contour = self.coordinate_transformer.get_contoured_image()
        plt.imshow(transformed_contour)
        plt.close()

    def test_rotation(self):
        self.coordinate_transformer.rotate_around_point()
        transformed_contour = self.coordinate_transformer.get_contoured_image()
        plt.imshow(transformed_contour)
        plt.close()

    def test_resize_points(self):
        self.coordinate_transformer.resize_points()
        transformed_contour = self.coordinate_transformer.get_contoured_image()
        plt.imshow(transformed_contour)
        plt.close()

    def test_composite_transformations(self):
        transformed_contour = self.coordinate_transformer.get_augmented_contoured_image()
        plt.imshow(transformed_contour)
        plt.close()