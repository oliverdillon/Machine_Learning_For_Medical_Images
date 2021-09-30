from pathlib import Path
import numpy as np
import math
from models.training_data import Training_data
from models.testing_data import Testing_data
class Process_and_save_feature_files:
    def __init__(self,dataset):
        self.allowed_organs = ["Right Parotid","Left Parotid"]
        self.ct_image_window = 300
        self.ct_image_level = 40
        self.image_width = self.image_height = 512
        self.filter_date = None
        self.dataset = dataset
        self.training_data = Training_data()
        self.testing_data = Testing_data()
        self.filter_patient_data()
        self.get_training_data()

    def is_contour_data(self,series):
        if series.modality =="RTSTRUCT":
            return True
        return False

    def is_image_data(self,series):
        if series.modality =="CT" and series.study_date == self.filter_date:
            return True
        return False

    def get_contour_data(self,patient):
        series_filtered =[]
        series_iterator = filter(self.is_contour_data,patient.series)
        for series in series_iterator:
            series_filtered.append(series)
            self.filter_date = series.study_date;
        # assert(series_filtered,len(1))
        return series_filtered

    def add_ct_images(self,patient,series_filtered):
        series_iterator = filter(self.is_image_data, patient.series)
        for series in series_iterator:
            series_filtered.append(series)

    def filter_patient_data(self):
        data = self.dataset.data
        for patient in data:
            series_filtered = self.get_contour_data(patient)
            self.add_ct_images(patient,series_filtered)

            patient.series = series_filtered

    def write_contour_to_image(self,Organ_Data,ImageArray,pydicom):
        for i in range(0,len(Organ_Data)-1):
            for j in range(0,len(Organ_Data[i])-1):
                if(Organ_Data[i][j][2] ==pydicom.ImagePositionPatient[2]):
                    OrganIndex1 = int((Organ_Data[i][j][0]-pydicom.ImagePositionPatient[0])/pydicom.PixelSpacing[0])
                    OrganIndex2 = int((Organ_Data[i][j][1]-pydicom.ImagePositionPatient[1])/pydicom.PixelSpacing[1])
                    ImageArray[OrganIndex2][OrganIndex1][2] = 255;

    def get_label(self,value):
        label = []
        for count, organ in enumerate(self.allowed_organs):
            label.append(0)
            if (value==organ):
                label[count] =1
        return label

    def overlay_contours(self,ct_image,organ):
        image = ct_image.dicomparser.GetImage(self.ct_image_window,self.ct_image_level)
        image = image.convert(mode ='RGB')
        ImageArray = np.asarray(image)
        ImageArray.flags.writeable = 1
        self.write_contour_to_image(organ,ImageArray,ct_image.pydicom)
        return ImageArray

    def create_directories(self,directory, features_array, labels_array):
        Path(directory).mkdir(parents=True, exist_ok=True)
        ct_image_directory = directory+"/features.npy"
        label_directory = directory+"/labels.txt"
        features_array.append(ct_image_directory)
        labels_array.append(label_directory)
        return ct_image_directory,label_directory

    def get_training_data(self):
        data = self.dataset.data
        total = len(data)
        testing_threshold = math.floor(0.6*total)

        for count, patient in enumerate(data):
            Organ_Dictionary = patient.series[0].contours_data
            Organ_Map = patient.series[0].organs
            ct_images = patient.series[1].medical_images

            if(count >= testing_threshold):
                directory = "target/training/"+patient.series[0].subject_ID
                ct_image_directory,label_directory = self.create_directories(directory, self.training_data.dataset, self.training_data.labels)
            else:
                directory = "target/testing/"+patient.series[0].subject_ID
                ct_image_directory,label_directory = self.create_directories(directory, self.testing_data.dataset, self.testing_data.labels)

            patient_contours = []
            patient_labels = []
            for key, value in Organ_Map.items():
                if value in self.allowed_organs:
                    ct_image_3d= []
                    label = self.get_label(value)
                    for ct_image in ct_images:
                        ImageArray = self.overlay_contours(ct_image,Organ_Dictionary[key])
                        ct_image_3d.append(ImageArray)
                    # self.plot_ct_image(ct_image_3d)
                    patient_contours.append(np.array(ct_image_3d))
                    patient_labels.append(label)
            if len(patient_contours)!= 0:
                np.save(ct_image_directory, np.array(patient_contours))
                np.savetxt(label_directory, np.array(patient_labels), delimiter=",")

        if len(self.testing_data.dataset)!= 0:
            np.savetxt("target/Training_features.txt", self.training_data.dataset, delimiter=",", fmt="%s")
            np.savetxt("target/Training_labels.txt", self.training_data.labels, delimiter=",", fmt="%s")
        if len(self.testing_data.dataset)!= 0:
            np.savetxt("target/Testing_features.txt", self.testing_data.dataset, delimiter=",", fmt="%s")
            np.savetxt("target/Testing_labels.txt", self.testing_data.labels, delimiter=",", fmt="%s")