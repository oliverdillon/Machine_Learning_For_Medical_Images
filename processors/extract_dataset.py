import csv
import operator

from models.series import Series
from models.patient import Patient
from models.dataset import Dataset

class Extract_dataset:
    def __init__(self,manifest_dir, batch_size):
        self.dataset =None
        self.manifest_dir = manifest_dir
        self.sort_csv_file()
        self.extract_dataset()

    def sort_csv_file(self):
        reader = csv.reader(open(self.manifest_dir))
        sortedlist = sorted(reader, key=operator.itemgetter(4), reverse=True)
        with open(self.manifest_dir, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for line in sortedlist:
                writer.writerow(line)

    def extract_dataset(self):
        print("Extracting Dataset")
        data = [];
        with open(self.manifest_dir, newline='') as csvfile:
            #TODO sort csv by patient id
            reader = csv.DictReader(csvfile)
            patient_id_state=""
            for count, row in enumerate(reader):
                series = Series(row)
                if (count==0):
                    patient = Patient(row)
                    patient_id_state=patient.subject_ID
                elif(patient_id_state!=series.subject_ID):
                    #TODO save
                    print("Extracted Data for "+patient.subject_ID)
                    data.append(patient);
                    patient = Patient(row)
                    patient_id_state=series.subject_ID
                patient.series.append(series)
            print("Extracted Data for "+patient.subject_ID)
            data.append(patient);
        self.dataset = Dataset(self.manifest_dir,data)