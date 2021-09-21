import csv
from models.series import Series
from models.patient import Patient
from models.dataset import Dataset

class Extract_dataset:
    def __init__(self,manifest_dir):
        data = [];
        with open(manifest_dir, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            patient_id_state=""
            for count, row in enumerate(reader):
                series = Series(row)
                if (count==0):
                    patient = Patient(row)
                    patient_id_state=patient.subject_ID
                elif(patient_id_state!=series.subject_ID):
                    data.append(patient);
                    patient = Patient(row)
                    patient_id_state=patient.subject_ID
                patient.series.append(series)
            data.append(patient);
        self.dataset = Dataset(manifest_dir,data)