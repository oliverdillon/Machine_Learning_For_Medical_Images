from pathlib import Path

from processors.process_and_save_feature_files import Process_and_save_feature_files
from processors.extract_dataset import Extract_dataset
from shutil import copyfile
import os.path
import re

class Batch_save_feature_files:
    def __init__(self, metadata_directory, save_directory, allowed_organs):
        self.allowed_organs = allowed_organs
        self.patient_batch_size = 2
        self.metadata_directory = metadata_directory
        self.save_directory = save_directory
        self.metadata_original_dir = self.metadata_directory\
            .replace("metadata", "/archive/metadata_original")
        self.store_original_metadata()
        self.extract_save_files()

    def store_original_metadata(self):
        if not(os.path.isfile(self.metadata_original_dir)):
            regex = re.compile(r"/metadata.+", re.IGNORECASE)
            directory = regex.sub("", self.metadata_original_dir)
            Path(directory).mkdir(parents=True, exist_ok=True)
            copyfile(self.metadata_directory, self.metadata_original_dir)

    def extract_save_files(self):
        extract_dataset = Extract_dataset(self.metadata_directory, self.patient_batch_size)
        Process_and_save_feature_files(extract_dataset.dataset, self.save_directory, self.allowed_organs)