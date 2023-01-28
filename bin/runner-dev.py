import os

from save_feature_file_batch_processor import SaveFeatureFileBatchProcessor
from n_fold_validator import NFoldValidator
from processed_medical_image_plotter import ProcessedMedicalImagePlotter

# https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries?Collection=HNSCC&format=html
count = 0
classified_organs = ["Right_Parotid", "Left_Parotid"]
augmented_organs = ["Right_Parotid", "Right_Parotid_augmented"]

while count < 1:
    SaveFeatureFileBatchProcessor(os.getcwd() + "/metadata_testing_classification.csv",
                                  os.getcwd() + "/target/classification", classified_organs)
    SaveFeatureFileBatchProcessor(os.getcwd() + "/metadata_testing_augmentation.csv",
                                  os.getcwd() + "/target/augmentation", augmented_organs)
    count += 1
# print("classification")
# ProcessedMedicalImagePlotter(os.getcwd()+"/target/classification", classified_organs)
# print("augmentation")
# ProcessedMedicalImagePlotter(os.getcwd()+"/target/augmentation", augmented_organs)
# NFoldValidator(classified_organs, 2, os.getcwd()+"/target/classification")
