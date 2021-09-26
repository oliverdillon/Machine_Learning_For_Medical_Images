import matplotlib.pyplot as plt
import numpy as np
from models.training_data import Training_data
from models.testing_data import Testing_data
class Overlay_contours_on_images:
    def __init__(self,dataset):
        self.allowed_organs = ["Right Parotid","Left Parotid"]
        self.ct_image_window = 300
        self.ct_image_level = 40
        self.filter_date =""
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

    def plot_ct_image(self, ct_image_3d):
        ct_image_3d =np.array(ct_image_3d,'uint8')
        overlaySagittalshape  = list(ct_image_3d[:,200,:,:].shape)
        overlaySagittal = ct_image_3d[:,200,:,:]
        plt.imshow(overlaySagittal)
        plt.close()

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

    def get_training_data(self):
        data = self.dataset.data
        total = len(data)
        testing_threshold = 0.6*total

        for count, patient in enumerate(data):
            Organ_Dictionary = patient.series[0].contours_data
            Organ_Map = patient.series[0].organs
            ct_images = patient.series[1].medical_images

            for key, value in Organ_Map.items():
                if value in self.allowed_organs:
                    ct_image_3d= []
                    label = self.get_label(value)

                    for ct_image in ct_images:
                        ImageArray = self.overlay_contours(ct_image,Organ_Dictionary[key])
                        ct_image_3d.append(ImageArray)
                    if(count > testing_threshold):
                        self.testing_data.dataset.append(ct_image_3d)
                        self.testing_data.labels.append(label)
                    else:
                        self.training_data.dataset.append(ct_image_3d)
                        self.training_data.labels.append(label)
                    self.plot_ct_image(ct_image_3d)
