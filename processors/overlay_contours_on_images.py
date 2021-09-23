import matplotlib.pyplot as plt
import numpy as np
class Overlay_contours_on_images:
    def __init__(self,dataset):
        self.filter_date =""
        self.dataset = dataset
        self.filter_patient_data()
        self.overlay_contours()

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

    def overlay_contours(self):
        data = self.dataset.data
        for patient in data:
            Organ_Dictionary = patient.series[0].contours_data
            Organ_Map = patient.series[0].organs;
            ct_images = patient.series[1].medical_images

            for key, value in Organ_Map.items():
                if value == "Brainstem":
                    Organ_Data = Organ_Dictionary[key]

                    for ct_image in ct_images:
                        # ImageArray = ct_image.pydicom.pixel_array
                        image = ct_image.dicomparser.GetImage(300,40)
                        image = image.convert(mode ='RGB')
                        ImageArray = np.asarray(image)
                        ImageArray.flags.writeable = 1
                        # plt.imshow(ImageArray)
                        # plt.title("Test")
                        # plt.show()
                        #Loops through contour data
                        for i in range(0,len(Organ_Data)-1):
                            for j in range(0,len(Organ_Data[i])-1):
                                    OrganIndex1 = int((Organ_Data[i][j][0]-ct_image.pydicom.ImagePositionPatient[0])/ct_image.pydicom.PixelSpacing[0])
                                    OrganIndex2 = int((Organ_Data[i][j][1]-ct_image.pydicom.ImagePositionPatient[1])/ct_image.pydicom.PixelSpacing[1])
                                    ImageArray[OrganIndex2][OrganIndex1][2] = 255;

                        # plt.imshow(ImageArray)
                        # plt.title("Test")
                        # plt.show()
