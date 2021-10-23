import re

import matplotlib.pyplot as plt
import numpy as np
import csv
class Plot_medical_images:
    def __init__(self, directory, allowed_organ):
        self.image_width = self.image_height = 512
        self.base_directory = directory
        self.allowed_organs = allowed_organ
        self.patient_class_1 = []
        self.patient_class_2 = []
        self.class_1 = []
        self.class_2 = []
        self.read_ct_images()
        plt.plot(self.patient_class_1, self.class_1, scaley='log')
        plt.plot(self.patient_class_2, self.class_2, scaley='log')
        plt.show()

    def read_ct_images(self):
        filename = self.base_directory+"/features.txt"
        with open(filename, 'r') as csvfile:
            feature_directories = csv.reader(csvfile)
            for directory in feature_directories:
                image = np.load(directory[0])
                patient_no_searcher = re.search('HNSCC-01-(.+?)/', directory[0])
                organ_no_searcher = re.search('(\w+)_feature.npy$', directory[0])
                if patient_no_searcher and organ_no_searcher:
                    patient_no = patient_no_searcher.group(1)
                    organ = organ_no_searcher.group(1)
                    if self.allowed_organs.index(organ) == 1:
                        self.patient_class_1.append(patient_no)
                        self.plot_ct_image(image, self.class_1)
                    else:
                        self.patient_class_2.append(patient_no)
                        self.plot_ct_image(image, self.class_2)

    def plot_ct_image(self, ct_image_3d, class_no):
        ct_image_3d_contour = ct_image_3d[..., 1]
        index = 0
        max_contour = 0
        total_contour_size = 0

        for i in range(self.image_width):
            contour_size = np.sum(ct_image_3d_contour[:, i, :])
            total_contour_size += contour_size
            if contour_size > max_contour:
                max_contour = contour_size
                index = i

        overlay_sagittal = ct_image_3d[:, index, :, :]
        plt.imshow(overlay_sagittal)
        plt.close()
        class_no.append(total_contour_size)
