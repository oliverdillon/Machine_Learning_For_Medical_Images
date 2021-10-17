import unittest
import transformation_functions
import matplotlib.pyplot as plt

class TestCTImage(unittest.TestCase):
    def setUp(self) :
        self.coordinates = []
        xcoordinates = [100,100,200,200]
        ycoordinates = [100,200,200,100]

        for i, ycoordinate in enumerate(ycoordinates):
            self.coordinates.append((xcoordinates[i],ycoordinate))

        before_translation = transformation_functions.FillContourArea(self.coordinates)
        plt.imshow(before_translation)
        plt.close()

    def test_translation(self):
        coordinates = transformation_functions.translate_points(self.coordinates)
        post_translation = transformation_functions.FillContourArea(coordinates)
        plt.imshow(post_translation)
        plt.close()

    def test_rotation(self):
        coordinates = transformation_functions.rotate_around_point(self.coordinates)
        post_translation = transformation_functions.FillContourArea(coordinates)
        plt.imshow(post_translation)
        plt.close()

    def test_random_points(self):
        coordinates = transformation_functions.MoveRandomPoints(self.coordinates, 0.1)
        post_translation = transformation_functions.FillContourArea(coordinates)
        plt.imshow(post_translation)
        plt.close()

    def test_resize_points(self):
        coordinates = transformation_functions.resize_points(self.coordinates)
        post_translation = transformation_functions.FillContourArea(coordinates)
        plt.imshow(post_translation)
        plt.close()

    def test_shear_points(self):
        coordinates = transformation_functions.shear_points(self.coordinates)
        post_translation = transformation_functions.FillContourArea(coordinates)
        plt.imshow(post_translation)
        plt.close()

    def test_composite_transformations(self):
        # post_translation = transformation_functions.translate_points(self.coordinates)
        post_translation = transformation_functions.move_random_points(self.coordinates, 0.1)
        post_translation = transformation_functions.shear_points(post_translation)
        post_translation = transformation_functions.rotate_around_point(post_translation)
        # post_translation = transformation_functions.resize_points(post_translation)
        post_translation = transformation_functions.FillContourArea(post_translation)
        plt.imshow(post_translation)
        plt.close()