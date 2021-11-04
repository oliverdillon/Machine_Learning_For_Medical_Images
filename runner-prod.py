from processors.batch_save_feature_files import Batch_save_feature_files

count = 0
classified_organs = ["Right_Parotid", "Left_Parotid"]
augmented_organs = ["Right_Parotid", "Right_Parotid_augmented"]

while count < 20:
    Batch_save_feature_files("D:/MPhys/manifest-1633461705263/metadata_classification.csv",
                             "D:/MPhys/classification", classified_organs)
    Batch_save_feature_files("D:/MPhys/manifest-1633461705263/metadata_augmentation.csv",
                             "D:/MPhys/augmentation", augmented_organs)
    count += 1