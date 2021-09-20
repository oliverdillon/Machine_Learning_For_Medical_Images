import glob
from ctimage import CTImage

class Series():
    def __init__(self,row):
        self.series_uid = row.get("Series UID")
        self.collection = row.get("Collection")
        self.study_description = row.get("Study Description")
        self.study_date = row.get("Study Date")
        self.modality = row.get("Modality")
        self.number_of_images = row.get("Number of Images")
        self.file_location = row.get("File Location")
        self.medical_images = self.extract_ct_images();

    def extract_ct_images(self):
        medical_images = []
        if self.modality =="CT":
            basepath = "C:/Users/Oliver/Documents/ME/Coding Projects/GitHub/Machine-Learning-For-Medical-Images"
            images_directories = glob.glob(basepath+self.file_location+"/*")
            for directory in images_directories:
                medical_images.append(CTImage(directory))
        return medical_images