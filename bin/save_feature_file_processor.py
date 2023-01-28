from pathlib import Path
import numpy as np
from processed_data_model import ProcessedDataModel
import scipy.ndimage

from util.file_functions import read_txt_and_append_to_list
from util.transformation_functions import Coordinate_transformer


def get_min_max_z_coordinate(organ_data, ct_images, max_z_location, min_z_location, reverse_factor):
    z_diff = 0
    for ct_image in ct_images:
        for i in range(0, len(organ_data) - 1):
            for j in range(0, len(organ_data[i]) - 1):
                if round(organ_data[i][j][2], 1) == round(reverse_factor * ct_image.zlocation, 1):
                    if abs(z_diff) < abs(ct_image.zlocation - min_z_location):
                        z_diff = ct_image.zlocation - min_z_location
                        max_z_location = ct_image.zlocation

    return min_z_location, max_z_location


def get_z_range(organ_dictionary, organ_map, ct_images):
    organ_data = organ_dictionary[organ_map["Brainstem"]]
    isocenter_data = organ_dictionary[organ_map["Isocenter"]]
    min_z_location = isocenter_data[0][0][2]
    max_z_location = None
    reverse_factor = 1
    min_z_location, max_z_location = get_min_max_z_coordinate(organ_data, ct_images,
                                                              max_z_location, min_z_location, reverse_factor)

    if max_z_location is None:
        # TODO Check if this is needed
        reverse_factor = -1
        min_z_location, max_z_location = get_min_max_z_coordinate(organ_data, ct_images,
                                                                  max_z_location, min_z_location, reverse_factor)

    if max_z_location is None:
        raise LookupError("Max Location is not found")
    else:
        return min_z_location, max_z_location, reverse_factor


def write_contour_to_image(organ_data, pydicom, reverse_factor):
    contour_points = []
    for i in range(0, len(organ_data) - 1):
        for j in range(0, len(organ_data[i]) - 1):
            if round(organ_data[i][j][2], 1) == round(reverse_factor * pydicom.ImagePositionPatient[2], 1):
                organ_index1 = int((organ_data[i][j][0] - pydicom.ImagePositionPatient[0]) / pydicom.PixelSpacing[0])
                organ_index2 = int((organ_data[i][j][1] - pydicom.ImagePositionPatient[1]) / pydicom.PixelSpacing[1])
                contour_points.append((organ_index1, organ_index2))
    return contour_points


def get_average_thickness(positions):
    array1 = positions[:-1]
    array2 = positions[1:]
    array3 = np.subtract(array1, array2)
    return abs(round(np.mean(array3), 1))


def create_directories(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    ct_image_directory = directory + "/{}_feature.npy"
    label_directory = directory + "/{}_label.txt"
    return ct_image_directory, label_directory


def is_contour_data(series):
    if series.modality == "RTSTRUCT":
        return True
    return False


def is_augmented(organ):
    if "augmented" in organ:
        return True
    return False


class SaveFeatureFileProcessor:
    def __init__(self, dataset, save_directory, allowed_organs):
        self.dataset = dataset
        self.save_base_directory = save_directory
        self.allowed_organs = allowed_organs
        self.required_contours = ["Right_Parotid", "Left_Parotid", "Isocenter", "Brainstem"]
        self.ct_image_window = 300
        self.ct_image_level = 40
        self.image_width = self.image_height = 512
        self.new_spacing = 3
        self.desired_matrix_length = 36
        self.filter_date = None
        self.processed_data = ProcessedDataModel()
        self.filter_patient_data()
        self.get_training_data()

    def get_contour_data(self, patient):
        series_filtered = []
        series_iterator = filter(is_contour_data, patient.series)
        for series in series_iterator:
            series_filtered.append(series)
            self.filter_date = series.study_date
        # assert(series_filtered,len(1))
        return series_filtered

    def is_image_data(self, series):
        if series.modality == "CT" and series.study_date == self.filter_date:
            return True
        return False

    def add_ct_images(self, patient, series_filtered):
        series_iterator = filter(self.is_image_data, patient.series)
        for series in series_iterator:
            series_filtered.append(series)

    def filter_patient_data(self):
        data = self.dataset.data
        for patient in data:
            series_filtered = self.get_contour_data(patient)
            self.add_ct_images(patient, series_filtered)

            patient.series = series_filtered

    def get_label(self, value):
        label = []
        for count, organ in enumerate(self.allowed_organs):
            label.append(0)
            if value == organ:
                label[count] = 1
        return label

    def overlay_contours(self, ct_image, organ, reverse_factor, is_augmented_value):
        image = ct_image.dicomparser.GetImage(self.ct_image_window, self.ct_image_level)
        image = image.convert(mode='RGB')
        image_array = np.asarray(image)
        image_array.flags.writeable = 1
        contour_points = write_contour_to_image(organ, ct_image.pydicom, reverse_factor)

        if len(contour_points) != 0:
            coordinate_transformer = Coordinate_transformer(contour_points)
            if is_augmented_value:
                image_array[..., 1] = coordinate_transformer.get_augmented_contoured_image()
            else:
                image_array[..., 1] = coordinate_transformer.get_contoured_image()
        else:
            image_array[..., 1] = np.zeros(image_array[..., 1].shape)

        return image_array

    def normalise_3d_image(self, ct_image_3d_dict):
        ct_image_3d = []
        z_locations = []

        # Correct the order of the array
        # ct_image_3d_dict = collections.OrderedDict(sorted(ct_image_3d_dict.items()))

        for key, value in ct_image_3d_dict.items():
            z_locations.append(key)
            ct_image_3d.append(value)

        thickness = get_average_thickness(z_locations)
        resize_factor = self.new_spacing / thickness

        new_ct_image_3d = scipy.ndimage.interpolation.zoom(ct_image_3d,
                                                           (resize_factor, 1, 1, 1), order=0, mode='nearest')

        while len(new_ct_image_3d) > self.desired_matrix_length:
            new_ct_image_3d = new_ct_image_3d[:-1]

        return new_ct_image_3d

    def get_training_data(self):
        data = self.dataset.data

        for count, patient in enumerate(data):
            if len(patient.series) == 0:
                print("Patient is missing contour data and/or image data")
                continue

            organ_dictionary = patient.series[0].contours_data
            organ_map = patient.series[0].organs
            ct_images = patient.series[1].medical_images
            subject_id = patient.series[0].subject_ID

            if any(required_contour not in organ_map.keys() for required_contour in self.required_contours):
                print("Patient " + subject_id + " did not have all the necessary contours")
                continue

            print("Saving for " + subject_id)
            directory = self.save_base_directory + "/" + subject_id
            ct_image_directory, label_directory = create_directories(directory)

            try:
                min_z_location, max_z_location, reverse_factor = get_z_range(organ_dictionary, organ_map, ct_images)
            except LookupError as err:
                print("Lookup Error: {0}".format(err))
                continue

            # loop through organ contours
            for key, value in organ_map.items():
                matches = [allowed_organ for allowed_organ in self.allowed_organs if key in allowed_organ]
                for organ in matches:
                    ct_image_3d_dict = {}
                    label = self.get_label(organ)
                    is_augmented_value = is_augmented(organ)

                    # build 3d image
                    for ct_image in ct_images:
                        if min_z_location < ct_image.zlocation < max_z_location:
                            image_array = self.overlay_contours(ct_image, organ_dictionary[value],
                                                                reverse_factor, is_augmented_value)
                            ct_image_3d_dict[ct_image.zlocation] = image_array

                    ct_image_3d = self.normalise_3d_image(ct_image_3d_dict)

                    # save matrix files
                    np.save(ct_image_directory.format(organ), np.array(ct_image_3d))
                    np.savetxt(label_directory.format(organ), np.array(label), delimiter=",")
                    self.processed_data.features.append(ct_image_directory.format(organ))
                    self.processed_data.labels.append(label_directory.format(organ))
            print("Saved for " + subject_id)

        # Save paths to files
        if len(self.processed_data.features) != 0:
            features_filename = self.save_base_directory + "/features.txt"
            labels_filename = self.save_base_directory + "/labels.txt"

            read_txt_and_append_to_list(features_filename, self.processed_data.features)
            read_txt_and_append_to_list(labels_filename, self.processed_data.labels)

            np.savetxt(features_filename, self.processed_data.features, delimiter=",", fmt="%s")
            np.savetxt(labels_filename, self.processed_data.labels, delimiter=",", fmt="%s")
