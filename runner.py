import os

from processors.batch_save_feature_files import Batch_save_feature_files
from processors.perform_n_fold_validation import Perform_n_fold_validation
from processors.plot_medical_images import Plot_medical_images
# from processors.train_neural_networks import Train_neural_network
#https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries?Collection=HNSCC&format=html
count = 0
classified_organs = ["Right_Parotid", "Left_Parotid"]
augmented_organs = ["Right_Parotid", "Right_Parotid_augmented"]

# while count< 7:
#     Batch_save_feature_files(os.getcwd()+"/metadata_testing_classification.csv",
#                                      os.getcwd()+"/target/classification", classified_organs)
#     Batch_save_feature_files(os.getcwd()+"/metadata_testing_augmentation.csv",
#                                      os.getcwd()+"/target/augmentation", augmented_organs)
#     count +=1
# print("classification")
Plot_medical_images(os.getcwd()+"/target/classification", classified_organs)
# print("augmentation")
# Plot_medical_images(os.getcwd()+"/target/augmentation", augmented_organs)
# Perform_n_fold_validation(classified_organs, 2, os.getcwd()+"/target/classification")
