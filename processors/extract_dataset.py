import csv
import operator

from models.series import Series
from models.patient import Patient
from models.dataset import Dataset

class Extract_dataset:
    def __init__(self,manifest_dir, batch_size):
        self.dataset =None
        self.manifest_dir = manifest_dir
        self.batch_size = batch_size
        self.extract_dataset()

    def convert_list_to_dict(self,list):
        dict =[]
        keys = list.pop(0)
        for row in list:
            dict.append({keys[i]:row[i]  for i in range(0, len(row))})
        list.insert(0,keys)
        return dict

    def extract_dataset(self):
        print("Extracting Dataset")
        data = []
        patient_id_state=""
        count = 1
        reader = csv.reader(open(self.manifest_dir))
        sortedlist = sorted(reader, key=operator.itemgetter(4), reverse=True)
        sortedDict = self.convert_list_to_dict(sortedlist)

        while count <= self.batch_size+1 and len(sortedDict)>0:
            sortedlist.pop()
            row = sortedDict.pop()
            series = Series(row)
            if (count==1):
                #declare first patient in file
                patient = Patient(row)
                patient_id_state=patient.subject_ID
                count+=1
            elif(patient_id_state!=series.subject_ID):
                #declare next patient
                patient = Patient(row)
                patient_id_state=series.subject_ID

            if(len(sortedDict)==0):
                #add last patient in file
                print("Extracted Data for "+patient.subject_ID)
                data.append(patient)
                count+=1
            elif(patient_id_state != sortedDict[len(sortedDict)-1]["Subject ID"]):
                #add next patient
                print("Extracted Data for "+patient.subject_ID)
                data.append(patient)
                count+=1
            patient.series.append(series)

        self.dataset = Dataset(self.manifest_dir,data)

        with open(self.manifest_dir, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for line in sortedlist:
                writer.writerow(line)