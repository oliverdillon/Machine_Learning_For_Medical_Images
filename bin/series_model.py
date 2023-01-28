import glob
from dicompylercore import dicomparser
from ct_image_model import CtImageModel


def extract_organ(index, dataset, contour_data, names, name, standardised_name, excluded_terms,
                  included_terms):
    data = []
    structure_coord_dicts = []
    if any(term in name for term in excluded_terms):
        return
    elif any(term in name for term in included_terms):
        structure_coord_dicts.append(dataset.GetStructureCoordinates(index))
        for j in list(structure_coord_dicts[0]):
            data.append(structure_coord_dicts[0][j][0]['data'])
        names[standardised_name] = name
        contour_data[name] = data


class SeriesModel:
    def __init__(self, root_directory, row):
        self.series_uid = row.get("Series UID")
        self.subject_ID = row.get("Subject ID")
        self.collection = row.get("Collection")
        self.study_description = row.get("Study Description")
        self.study_date = row.get("Study Date")
        self.modality = row.get("Modality")
        self.number_of_images = row.get("Number of Images")
        self.file_location = row.get("File Location")
        self.root_directory = root_directory
        self.medical_images = self.extract_ct_images()
        self.contoured_medical_images = []
        self.contours_data, self.organs = self.extract_contour_data()

    def extract_ct_images(self):
        medical_images = []
        if self.modality == 'CT':
            images_directories = glob.glob(self.root_directory + self.file_location + "/*")
            for directory in images_directories:
                medical_images.append(CtImageModel(directory))
        return medical_images

    def extract_contour_data(self):
        contour_data = {}
        names = {}
        if self.modality == 'RTSTRUCT':
            structure_directories = glob.glob(self.root_directory + self.file_location + "/*")
            dataset = dicomparser.DicomParser(structure_directories[0])
            structures = dataset.GetStructures()

            for index in range(1, len(structures)):
                try:
                    name = str(structures[index]['name']).lower()

                    if "brain" in name:
                        excluded_terms = ["ex", "2", "cm", "mm", "pv"]
                        included_terms = ["brainstem", "brain stem"]

                        extract_organ(index, dataset, contour_data, names, name, "Brainstem", excluded_terms,
                                      included_terms)

                    if "parotid" in name or "prtd" in name:
                        excluded_terms = ["sub", "total", "def", "sup", "deep", "gy", "avoid", "ptv", "push", "tail"]
                        included_terms_right = ["rt", "r ", " r", "right"]
                        included_terms_left = ["lt", "l ", " l", "left"]

                        extract_organ(index, dataset, contour_data, names, name, "Right_Parotid", excluded_terms,
                                      included_terms_right)
                        extract_organ(index, dataset, contour_data, names, name, "Left_Parotid", excluded_terms,
                                      included_terms_left)

                    if "ring" in name:
                        excluded_terms = ["inner", "ext"]
                        included_terms = ["ring"]

                        extract_organ(index, dataset, contour_data, names, name, "Ring_Boundary", excluded_terms,
                                      included_terms)

                    if "external" in name:
                        excluded_terms = []
                        included_terms = ["external"]

                        extract_organ(index, dataset, contour_data, names, name, "External_Boundary",
                                      excluded_terms, included_terms)

                    if "iso" in name or "mark" in name:
                        excluded_terms = ["final", "lao", "boost"]
                        included_terms = ["iso", "mark"]

                        extract_organ(index, dataset, contour_data, names, name, "Isocenter", excluded_terms,
                                      included_terms)

                    if "cochlea" in name:
                        excluded_terms = ["sub", "total", "def", "sup", "deep", "gy", "avoid", "ptv", "push", "tail"]
                        included_terms_right = ["rt", "r ", " r", "right"]
                        included_terms_left = ["lt", "l ", " l", "left"]

                        extract_organ(index, dataset, contour_data, names, name, "Right_Cochlea", excluded_terms,
                                      included_terms_right)
                        extract_organ(index, dataset, contour_data, names, name, "Left_Cochlea", excluded_terms,
                                      included_terms_left)
                except Exception as e:
                    print("Exception: {} raised at Index {}".format(e, index))

        return contour_data, names
