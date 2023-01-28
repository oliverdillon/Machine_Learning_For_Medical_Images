from save_feature_file_batch_processor import SaveFeatureFileBatchProcessor

count = 0
classified_organs = ["Right_Parotid", "Left_Parotid"]
augmented_organs = ["Right_Parotid", "Right_Parotid_augmented"]

while count < 20:
    SaveFeatureFileBatchProcessor("D:/MPhys/manifest-1633461705263/metadata_classification.csv",
                                  "D:/MPhys/classification", classified_organs)
    SaveFeatureFileBatchProcessor("D:/MPhys/manifest-1633461705263/metadata_augmentation.csv",
                                  "D:/MPhys/augmentation", augmented_organs)
    count += 1
