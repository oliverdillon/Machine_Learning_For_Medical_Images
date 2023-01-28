from pathlib import Path

from save_feature_file_processor import SaveFeatureFileProcessor
from dataset_extractor import DatasetExtractor
from shutil import copyfile
import os.path
import re


class SaveFeatureFileBatchProcessor:
    def __init__(self, metadata_directory, save_directory, allowed_organs, patient_batch_size=2):
        print("STARTED - SaveFeatureFileBatchProcessor with parameters: "
              "\nmetadata_directory: {},"
              "\nsave_directory: {},"
              "\nallowed_organs: {}".format(metadata_directory, save_directory, allowed_organs))
        self.allowed_organs = allowed_organs
        self.patient_batch_size = patient_batch_size
        self.metadata_directory = metadata_directory
        self.save_directory = save_directory
        self.metadata_original_dir = self.metadata_directory \
            .replace("metadata", "/archive/metadata_original")
        regex = re.compile(r"/metadata.+", re.IGNORECASE)
        self.root_archive_directory = regex.sub("", self.metadata_original_dir)
        self.root_directory = regex.sub("", self.metadata_directory)
        self.store_original_metadata()
        self.extract_save_files()
        print("COMPLETE - SaveFeatureFileBatchProcessor")

    def store_original_metadata(self):
        if not (os.path.isfile(self.metadata_original_dir)):
            Path(self.root_archive_directory).mkdir(parents=True, exist_ok=True)
            copyfile(self.metadata_directory, self.metadata_original_dir)

    def extract_save_files(self):
        extract_dataset = DatasetExtractor(self.root_directory, self.metadata_directory, self.patient_batch_size)
        SaveFeatureFileProcessor(extract_dataset.dataset, self.save_directory, self.allowed_organs)
