from models.series import Series
from models.patient import Patient
from models.dataset import Dataset
from util.file_functions import read_csv_as_sorted_list, save_csv_list


class Extract_dataset:
    def __init__(self, metadata_dir, batch_size):
        self.dataset = None
        self.metadata_dir = metadata_dir
        self.metadata_processed_dir = self.metadata_dir.replace("metadata", "metadata_processed")
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
        patient_id_state = None
        count = 1
        sorted_metadata_list = read_csv_as_sorted_list(self.metadata_dir)
        sorted_metadata_dict = self.convert_list_to_dict(sorted_metadata_list)
        sorted_processed_metadata_list = read_csv_as_sorted_list(self.metadata_processed_dir)

        while count <= self.batch_size+1 and len(sorted_metadata_dict) > 0:
            metadata_list_row = sorted_metadata_list.pop()
            sorted_processed_metadata_list.append(metadata_list_row)
            metadata_dict_row = sorted_metadata_dict.pop()
            series = Series(metadata_dict_row)
            if count == 1:
                # declare first patient in file
                patient = Patient(metadata_dict_row)
                patient_id_state = patient.subject_ID
                count += 1
            elif patient_id_state != series.subject_ID:
                # declare next patient
                patient = Patient(metadata_dict_row)
                patient_id_state = series.subject_ID

            if len(sorted_metadata_dict) == 0:
                # add last patient in file
                print("Extracted Data for " + patient.subject_ID)
                data.append(patient)
                count += 1
            elif patient_id_state != sorted_metadata_dict[len(sorted_metadata_dict)-1]["Subject ID"]:
                # add next patient
                print("Extracted Data for "+patient.subject_ID)
                data.append(patient)
                count += 1
            patient.series.append(series)

        self.dataset = Dataset(self.metadata_dir, data)
        save_csv_list(self.metadata_processed_dir, sorted_processed_metadata_list)
        save_csv_list(self.metadata_dir, sorted_metadata_list)
