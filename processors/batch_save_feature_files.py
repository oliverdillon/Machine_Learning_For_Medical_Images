from processors.process_and_save_feature_files import Process_and_save_feature_files
from processors.extract_dataset import Extract_dataset

class Batch_save_feature_files:
    def __init__(self,extract_directory,save_directory):
        self.patient_batch_size= 2
        self.extract_directory = extract_directory
        self.save_directory = save_directory
        self.extract_save_files()

    def extract_save_files(self):
        extract_dataset = Extract_dataset(self.extract_directory, self.patient_batch_size)
        filtered_dataset = Process_and_save_feature_files(self.save_directory, extract_dataset.dataset)