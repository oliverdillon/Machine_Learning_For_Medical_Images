from processors.process_and_save_feature_files import Process_and_save_feature_files
from processors.extract_dataset import Extract_dataset
import os
class Batch_save_feature_files:
    def __init__(self,directory):
        self.patient_batch_size= 2
        self.directory = directory
        self.extract_save_files()

    def extract_save_files(self):
        extract_dataset = Extract_dataset(self.directory, self.patient_batch_size)
        filtered_dataset = Process_and_save_feature_files(extract_dataset.dataset)