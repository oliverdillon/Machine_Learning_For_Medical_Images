import matplotlib.pyplot as plt
import numpy as np
import csv
class Plot_medical_images:
    def __init__(self):
        self.read_ct_images()

    def read_ct_images(self):
        filename = "target/Training_features.txt"
        with open(filename, 'r') as csvfile:
            training_feature_directories = csv.reader(csvfile)
            for directory in training_feature_directories:
                images = np.load(directory[0])
                for image in images:
                    self.plot_ct_image(image)

    def plot_ct_image(self, ct_image_3d):
        ct_image_3d =np.array(ct_image_3d,'uint8')
        overlaySagittalshape  = list(ct_image_3d[:,200,:,:].shape)
        overlaySagittal = ct_image_3d[:,200,:,:]
        plt.imshow(overlaySagittal)
        plt.close()