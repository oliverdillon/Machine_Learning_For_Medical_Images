# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:33:11 2020

@author: Oliver
"""
import random
import math
import numpy as np
from PIL import Image, ImageDraw


class Coordinate_transformer:
    def __init__(self, contour_points):
        self.x, self.y, self.ox, self.oy = self.get_coordinates(contour_points)
        self.contour_points = contour_points

    def get_coordinates(self, points):
        x = [point[0] for point in points]
        y = [point[1] for point in points]

        ox = sum(x) / len(x)
        oy = sum(y) / len(y)

        return x, y, ox, oy

    def get_contoured_images(self):
        # http://stackoverflow.com/a/3732128/1410871
        img = Image.new(mode='L', size=(512, 512), color=0)
        ImageDraw.Draw(img).polygon(xy=self.contour_points, outline=0, fill=1)
        # img = img.transpose(Image.ROTATE_90)
        mask = np.array(img).astype(bool)

        return np.uint8(mask) * 255

    def translate_contour(self):
        contour_points = []

        x_translation = random.uniform(2, 5)
        y_translation = random.uniform(2, 5)

        x_translation = x_translation * (-1) ** random.randint(1, 2)
        y_translation = y_translation * (-1) ** random.randint(1, 2)

        for i in range(len(self.x)):
            qx = self.x[i] + x_translation
            qy = self.y[i] + y_translation
            contour_points.append((qx, qy))

        self.contour_points = contour_points

    def move_random_points(self, percentage):
        contour_points = []
        indices = []
        i = 0
        number_of_points = len(self.x)

        while (i < number_of_points * percentage):
            random_index = random.randint(0, number_of_points - 1)
            random_cluster = random.randint(random_index + 1, number_of_points)
            for k in range(random_index, random_cluster):
                if not (k in indices):
                    indices.append(k)
                    i = i + 1

        for j in indices:
            x_pixel_translation = random.uniform(2, 5) * (-1) ** random.randint(1, 2)
            y_pixel_translation = random.uniform(2, 5) * (-1) ** random.randint(1, 2)

            self.x[j] = self.x[j] + x_pixel_translation
            self.y[j] = self.y[j] + y_pixel_translation

        for i in range(len(self.x)):
            contour_points.append((self.x[i], self.y[i]))

        self.contour_points = contour_points

    def shear_points(self):
        dx = [(point - self.ox) for point in self.x]
        dy = [(point - self.oy) for point in self.y]

        phi = random.uniform(88 , 90) * math.pi / 180
        M = 1.0 / np.tan(phi)

        XorY = random.randint(0, 1)
        contour_points = []

        if XorY == 0:
            for i in range(len(self.x)):
                qx = self.x[i] + (M * dy[i])
                qy = self.y[i]
                contour_points.append((qx, qy))

        else:
            for i in range(len(self.y)):
                qx = self.x[i]
                qy = self.y[i] + (M * dx[i])
                contour_points.append((qx, qy))

        self.contour_points = contour_points

    def rotate_around_point(self):
        contour_points = []

        radians = random.uniform(0, 5)
        radians = radians * (-1) ** random.randint(1, 2)
        radians = radians * math.pi / 180

        for i in range(len(self.x)):
            qx = self.ox + math.cos(radians) * (self.x[i] - self.ox) + math.sin(radians) * (self.y[i] - self.oy)
            qy = self.oy + -math.sin(radians) * (self.x[i] - self.ox) + math.cos(radians) * (self.y[i] - self.oy)
            contour_points.append((qx, qy))

        self.contour_points = contour_points

    def resize_points(self):
        scaling_factor = random.randrange(9, 11, 1) / 10
        contour_points = []
        r = []
        theta = []

        dx = [(point - self.ox) for point in self.x]
        dy = [(point - self.oy) for point in self.y]

        for i in range(len(self.x)):
            r.append(np.sqrt(math.pow(dx[i], 2) + math.pow(dy[i], 2)))
            theta.append(np.arctan2(dy[i], dx[i]))

        for i in range(len(self.x)):
            qx = self.ox + (scaling_factor * r[i] * np.sin(theta[i]))
            qy = self.oy + (scaling_factor * r[i] * np.cos(theta[i]))
            contour_points.append((qx, qy))

        self.contour_points = contour_points