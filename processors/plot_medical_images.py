import matplotlib.pyplot as plt
import numpy as np
import csv
class Plot_medical_images:
    def __init__(self, directory):
        self.image_width = self.image_height = 512
        self.base_directory = directory
        self.read_ct_images()

    def read_ct_images(self):
        filename = self.base_directory+"/features.txt"
        with open(filename, 'r') as csvfile:
            feature_directories = csv.reader(csvfile)
            for directory in feature_directories:
                image = np.load(directory[0])
                self.plot_ct_image(image)

    def plot_ct_image(self, ct_image_3d):
        ct_image_3d_contour =ct_image_3d[...,1]
        index = 0
        while(index < self.image_width and (np.sum(ct_image_3d_contour[:,index,:])==0 or np.sum(ct_image_3d_contour[:,index,:])<np.sum(ct_image_3d_contour[:,index+1,:]) )):
            index+=1
        overlaySagittal = ct_image_3d[:,index,:,:]
        plt.imshow(overlaySagittal)
        plt.close()