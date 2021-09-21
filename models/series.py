import glob
import os
from dicompylercore import dicomparser
from models.ctimage import CTImage

class Series():
    def __init__(self,row):
        self.series_uid = row.get("Series UID")
        self.subject_ID= row.get("Subject ID")
        self.collection = row.get("Collection")
        self.study_description = row.get("Study Description")
        self.study_date = row.get("Study Date")
        self.modality = row.get("Modality")
        self.number_of_images = row.get("Number of Images")
        self.file_location = row.get("File Location")
        self.medical_images = self.extract_ct_images();
        self.contours_data, self.organs = self.extract_contour_data();

    def extract_ct_images(self):
        medical_images = []
        if self.modality =="CT":
            images_directories = glob.glob(os.getcwd()+self.file_location+"/*")
            for directory in images_directories:
                medical_images.append(CTImage(directory))
        return medical_images

    def extract_contour_data(self):
        contour_data={}
        names = {}
        if self.modality =="RTSTRUCT":
            images_directories = glob.glob(os.getcwd()+self.file_location+"/*")
            dataset = dicomparser.DicomParser(images_directories[0])
            structures = dataset.GetStructures()

            for index in range(1,len(structures)): #to strip out the parotids
                try:
                    name = str(structures[index]['name']).lower()

                    if "brainstem" in name:
                        exluded_terms = ["ex","2","cm","mm","pv"]
                        included_terms = ["brainstem","brain stem"]

                        self.extract_organ(index,dataset,contour_data,names,name,"Brainstem",exluded_terms,included_terms)

                    if "parotid" in name or "prtd" in name:
                        exluded_terms = ["sub","total","def","sup","deep","gy","avoid","ptv","push","tail"]
                        included_terms_right = ["rt","r "," r","right"]
                        included_terms_left = ["lt","l "," l","left"]

                        self.extract_organ(index,dataset,contour_data,names,name,"Right Parotid",exluded_terms,included_terms_right)
                        self.extract_organ(index,dataset,contour_data,names,name,"Left Parotid",exluded_terms,included_terms_left)

                    if "ring" in name:
                        exluded_terms = ["inner","ext"]
                        included_terms = ["ring"]

                        self.extract_organ(index,dataset,contour_data,names,name,"Ring Boundary",exluded_terms,included_terms)

                    if "external" in name:
                        exluded_terms = []
                        included_terms = ["external"]

                        self.extract_organ(index,dataset,contour_data,names,name,"External Boundary",exluded_terms,included_terms)

                    if "iso" in name:
                        exluded_terms = ["final","lao","boost"]
                        included_terms = ["mark"]

                        self.extract_organ(index,dataset,contour_data,names,name,"External Boundary",exluded_terms,included_terms)

                    if "cochlea" in name:
                        exluded_terms = ["sub","total","def","sup","deep","gy","avoid","ptv","push","tail"]
                        included_terms_right = ["rt","r "," r","right"]
                        included_terms_left = ["lt","l "," l","left"]

                        self.extract_organ(index,dataset,contour_data,names,name,"Right Cochlea",exluded_terms,included_terms_right)
                        self.extract_organ(index,dataset,contour_data,names,name,"Left Cochlea",exluded_terms,included_terms_left)
                except:
                    print("Error at Index %2i"%index)

        return contour_data, names

    def extract_organ(self,index,dataset,contour_data, names, name, standardised_name, exluded_terms, included_terms):
        data = []
        StructureCoordDicts =[]
        if any(term in name for term in exluded_terms) :
            return
        elif any(term in name for term in included_terms):
            StructureCoordDicts.append(dataset.GetStructureCoordinates(index))
            for j in list(StructureCoordDicts[0]): #iterate through the dictionary i get in line 14
                data.append(StructureCoordDicts[0][j][0]['data']) #pull out only the matrix of xyz values
            names[name] = standardised_name
            contour_data[name] = data