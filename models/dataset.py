import csv
from models.series import Series
from models.patient import Patient

class Dataset:
    def __init__(self,manifest_dir):
        self.manifest = self.extract_manifest(manifest_dir)


    def extract_manifest(self,manifest_dir):
        manifest = [];
        with open(manifest_dir, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            patient_id_state=""
            for count, row in enumerate(reader):
                series = Series(row)
                if (count==0):
                    patient = Patient(row)
                    patient_id_state=patient.subject_ID
                elif(patient_id_state!=series.subject_ID):
                    manifest.append(patient);
                    patient = Patient(row)
                    patient_id_state=patient.subject_ID
                patient.series.append(series)
            manifest.append(patient);
        return manifest;