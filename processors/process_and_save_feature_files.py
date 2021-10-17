from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from models.processed_data import Processed_data
import scipy.ndimage
import csv
class Process_and_save_feature_files:
    def __init__(self, save_directory, dataset):
        self.allowed_organs = ["Right_Parotid","Left_Parotid"]
        self.required_contours = ["Right_Parotid","Left_Parotid","Isocenter", "Brainstem"]
        self.save_base_directory = save_directory
        self.ct_image_window = 300
        self.ct_image_level = 40
        self.image_width = self.image_height = 512
        self.new_spacing = 3
        self.desired_matrix_length = 36
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

    def write_contour_to_image(self,Organ_Data,pydicom, reverse_factor):
        countour_points = []
        for i in range(0,len(Organ_Data)-1):
            for j in range(0,len(Organ_Data[i])-1):
                if(round(Organ_Data[i][j][2],1) == round(reverse_factor*pydicom.ImagePositionPatient[2],1)):
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

    def overlay_contours(self,ct_image,organ, reverse_factor):
        image = ct_image.dicomparser.GetImage(self.ct_image_window,self.ct_image_level)
        image = image.convert(mode ='RGB')
        ImageArray = np.asarray(image)
        ImageArray.flags.writeable = 1
        contour_points = self.write_contour_to_image(organ,ct_image.pydicom,reverse_factor)

        if len(contour_points)!= 0:
            ImageArray[..., 1] = self.fill_contour_area(contour_points)
        else:
            ImageArray[..., 1] = np.zeros(ImageArray[..., 1].shape)

        return ImageArray

    def average_thickness(self, positions):
        array1 = positions[:-1]
        array2 = positions[1:]
        array3 = np.subtract(array1,array2)
        return abs(round(np.mean(array3),1))

    def normalise_3d_image(self,ct_image_3d_dict):
        ct_image_3d = []
        z_locations = []

        #Correct the order of the array
        # ct_image_3d_dict = collections.OrderedDict(sorted(ct_image_3d_dict.items()))

        for key, value in ct_image_3d_dict.items():
            z_locations.append(key)
            ct_image_3d.append(value)

        thickness = self.average_thickness(z_locations)
        resize_factor = self.new_spacing/thickness

        new_ct_image_3d = scipy.ndimage.interpolation.zoom(ct_image_3d,(resize_factor,1,1,1), order=0, mode='nearest')

        while len(new_ct_image_3d) >  self.desired_matrix_length:
            new_ct_image_3d = new_ct_image_3d[:-1]

        return new_ct_image_3d

    def get_min_max_z_coordinate(self,organ_data,ct_images,max_z_location,min_z_location,reverse_factor):
        z_diff = 0
        for ct_image in ct_images:
            z_location = ct_image.pydicom.ImagePositionPatient[2]
            for i in range(0,len(organ_data)-1):
                for j in range(0,len(organ_data[i])-1):
                    if(round(organ_data[i][j][2],1) == round(reverse_factor*z_location,1)):
                        if (abs(z_diff) < abs(z_location-min_z_location)):
                            z_diff = z_location-min_z_location
                            max_z_location = z_location

        return min_z_location, max_z_location

    def get_z_range(self,organ_dictionary,organ_map,ct_images):
        organ_data = organ_dictionary[organ_map["Brainstem"]]
        isocenter_data = organ_dictionary[organ_map["Isocenter"]]
        min_z_location = isocenter_data[0][0][2]
        max_z_location = None
        reverse_factor = 1
        min_z_location, max_z_location = self.get_min_max_z_coordinate(organ_data,ct_images,max_z_location,min_z_location,reverse_factor)

        if (max_z_location == None):
            #TODO Check if this is needed
            reverse_factor = -1
            min_z_location, max_z_location = self.get_min_max_z_coordinate(organ_data,ct_images,max_z_location,min_z_location,reverse_factor)

        if (max_z_location == None):
            return
        else:
            return min_z_location, max_z_location, reverse_factor

    def create_directories(self,directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        ct_image_directory = directory+"/{}_feature.npy"
        label_directory = directory+"/{}_label.txt"
        return ct_image_directory,label_directory

    def get_training_data(self):
        data = self.dataset.data

        for count, patient in enumerate(data):
            organ_dictionary = patient.series[0].contours_data
            organ_map = patient.series[0].organs
            ct_images = patient.series[1].medical_images
            subject_ID = patient.series[0].subject_ID

            if any(required_contour not in organ_map.keys() for required_contour in self.required_contours):
                print("Patient "+subject_ID+" did not have all the necessary contours")
                continue

            print ("Saving for "+subject_ID)
            directory = self.save_base_directory+"/"+subject_ID
            ct_image_directory,label_directory = self.create_directories(directory)

            min_z_location, max_z_location, reverse_factor = self.get_z_range(organ_dictionary,organ_map,ct_images)

            #loop through organ contours
            for key, value  in organ_map.items():
                if key in self.allowed_organs:
                    ct_image_3d_dict= {}
                    label = self.get_label(key)

                    #build 3d image
                    for ct_image in ct_images:
                        z_location = ct_image.pydicom.ImagePositionPatient[2]

                        if(min_z_location < z_location < max_z_location):
                            ImageArray = self.overlay_contours(ct_image,organ_dictionary[value],reverse_factor)
                            ct_image_3d_dict[z_location] = ImageArray

                    ct_image_3d = self.normalise_3d_image(ct_image_3d_dict)

                    #save matrix files
                    np.save(ct_image_directory.format(key), np.array(ct_image_3d))
                    np.savetxt(label_directory.format(key), np.array(label), delimiter=",")
                    self.processed_data.features.append(ct_image_directory.format(key))
                    self.processed_data.labels.append(label_directory.format(key))

        #Save paths to files
        if len(self.processed_data.features)!= 0:
            features_filename = self.save_base_directory+"/features.txt"
            labels_filename = self.save_base_directory+"/labels.txt"
            try:
                with open(features_filename, 'r') as csvfile:
                    feature_reader = csv.reader(csvfile)
                    for directory in feature_reader:
                        self.processed_data.features.append(directory[0])

                with open(labels_filename, 'r') as csvfile:
                    labels_reader = csv.reader(csvfile)
                    for directory in labels_reader:
                        self.processed_data.labels.append(directory[0])
            except:
                print("Error opening directories file")

            np.savetxt(features_filename, self.processed_data.features, delimiter=",", fmt="%s")
            np.savetxt(labels_filename, self.processed_data.labels, delimiter=",", fmt="%s")