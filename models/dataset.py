import csv
from models.series import Series

class Dataset:
    def __init__(self,manifest_dir):
        self.manifest = self.extract_manifest(manifest_dir)

    def extract_manifest(self,manifest_dir):
        manifest = [];
        with open(manifest_dir, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                manifest.append(Series(row));
        return manifest;