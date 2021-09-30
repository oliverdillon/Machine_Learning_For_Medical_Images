from processors.overlay_contours_on_images import Overlay_contours_on_images
from processors.extract_dataset import Extract_dataset
from processors.plot_medical_images import Plot_medical_images
# from processors.train_neural_networks import Train_neural_network
import os

# Extract_dataset = Extract_dataset(os.getcwd()+"/metadata.csv")
# filtered_dataset = Overlay_contours_on_images(Extract_dataset.dataset)

run = Plot_medical_images()
# index1 = 0
# index2 = int(round(len(filtered_dataset.training_data)*0.1,0))-1
# train_neural_network= Train_neural_network(filtered_dataset.training_data,filtered_dataset.testing_data,index1,index2)
