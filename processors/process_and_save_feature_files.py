from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from models.processed_data import Processed_data
class Process_and_save_feature_files:
    def __init__(self,dataset):
        self.allowed_organs = ["Right_Parotid","Left_Parotid"]
        self.ct_image_window = 300
        self.ct_image_level = 40
        self.image_width = self.image_height = 512
        self.filter_date = None
        self.dataset = dataset
        self.processed_data = Processed_data()
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

    def write_contour_to_image(self,Organ_Data,pydicom):
        countour_points = []
        for i in range(0,len(Organ_Data)-1):
            for j in range(0,len(Organ_Data[i])-1):
                if(Organ_Data[i][j][2] ==pydicom.ImagePositionPatient[2]):
                    OrganIndex1 = int((Organ_Data[i][j][0]-pydicom.ImagePositionPatient[0])/pydicom.PixelSpacing[0])
                    OrganIndex2 = int((Organ_Data[i][j][1]-pydicom.ImagePositionPatient[1])/pydicom.PixelSpacing[1])
                    countour_points.append((OrganIndex1,OrganIndex2))
        return countour_points

    def get_label(self,value):
        label = []
        for count, organ in enumerate(self.allowed_organs):
            label.append(0)
            if (value==organ):
                label[count] =1
        return label

    def fill_contour_area(self,Vertices):
        # http://stackoverflow.com/a/3732128/1410871
        img = Image.new(mode='L', size=(self.image_width,self.image_height), color=0)
        ImageDraw.Draw(img).polygon(xy=Vertices, outline=0, fill=1)
        #img = img.transpose(Image.ROTATE_90)
        mask = np.array(img).astype(bool)

        return np.uint8(mask)*255

    def overlay_contours(self,ct_image,organ):
        image = ct_image.dicomparser.GetImage(self.ct_image_window,self.ct_image_level)
        image = image.convert(mode ='RGB')
        ImageArray = np.asarray(image)
        ImageArray.flags.writeable = 1
        contour_points = self.write_contour_to_image(organ,ct_image.pydicom)

        if len(contour_points)!= 0:
            ImageArray[..., 1] = self.fill_contour_area(contour_points)
        else:
            ImageArray[..., 1] = np.zeros(ImageArray[..., 1].shape)

        return ImageArray

    def create_directories(self,directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        ct_image_directory = directory+"/{}_feature.npy"
        label_directory = directory+"/{}_label.txt"
        return ct_image_directory,label_directory

    def get_training_data(self):
        data = self.dataset.data

        for count, patient in enumerate(data):
            Organ_Dictionary = patient.series[0].contours_data
            Organ_Map = patient.series[0].organs
            ct_images = patient.series[1].medical_images
            subject_ID = patient.series[0].subject_ID
            print ("Saving for "+subject_ID)
            directory = "target/"+subject_ID
            ct_image_directory,label_directory = self.create_directories(directory)

            #loop through organ contours
            for key, value in Organ_Map.items():
                if value in self.allowed_organs:
                    ct_image_3d= []
                    label = self.get_label(value)

                    #build 3d image
                    for ct_image in ct_images:
                        ImageArray = self.overlay_contours(ct_image,Organ_Dictionary[key])
                        ct_image_3d.append(ImageArray)

                    #save matrix files
                    np.save(ct_image_directory.format(value), np.array(ct_image_3d))
                    np.savetxt(label_directory.format(value), np.array(label), delimiter=",")
                    self.processed_data.features.append(ct_image_directory.format(value))
                    self.processed_data.labels.append(label_directory.format(value))

        #Save paths to files
        if len(self.processed_data.features)!= 0:
            np.savetxt("target/features.txt", self.processed_data.features, delimiter=",", fmt="%s")
            np.savetxt("target/labels.txt", self.processed_data.labels, delimiter=",", fmt="%s")