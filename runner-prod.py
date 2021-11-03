from processors.batch_save_feature_files import Batch_save_feature_files

count = 0
classified_organs = ["Right_Parotid", "Left_Parotid"]
augmented_organs = ["Right_Parotid", "Right_Parotid_augmented"]

while count< 7:
    Batch_save_feature_files("D:/Masters/manifest-1634647071762/metadata.csv",
                             "D:/Masters/classification", classified_organs)
    Batch_save_feature_files("D:/Masters/manifest-1634647071762/metadata.csv",
                             "D:/Masters//augmentation", augmented_organs)
    count +=1