import os

from processors.batch_save_feature_files import Batch_save_feature_files
from processors.plot_medical_images import Plot_medical_images
# from processors.train_neural_networks import Train_neural_network
#https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries?Collection=HNSCC&format=html
count = 0
while count< 7:
    batch = Batch_save_feature_files(os.getcwd()+"/metadata.csv")
    count +=1
run = Plot_medical_images()
# index1 = 0
# index2 = int(round(len(filtered_dataset.training_data)*0.1,0))-1
# train_neural_network= Train_neural_network(filtered_dataset.training_data,filtered_dataset.testing_data,index1,index2)
