#!/usr/bin/env python
"""
Script that creates an object to

"""
import argparse
# import os

from series_model import SeriesModel
from patient_model import PatientModel
from dataset_model import DatasetModel
from util.file_functions import read_csv_as_sorted_list, save_csv_list


def convert_list_to_dict(list_param):
    dictionary = []
    keys = list_param.pop(0)
    for row in list_param:
        dictionary.append({keys[i]: row[i] for i in range(0, len(row))})
    list_param.insert(0, keys)
    return dictionary


class DatasetExtractor:
    def __init__(self, root_directory, metadata_dir, batch_size):
        self.dataset = None
        self.metadata_dir = metadata_dir
        self.metadata_processed_dir = self.metadata_dir \
            .replace("metadata", "archive/metadata_processed")
        self.root_directory = root_directory
        self.batch_size = batch_size
        self.extract_dataset()

    def extract_dataset(self):
        print("Extracting Dataset")
        data = []
        patient_id_state = None
        count = 1
        sorted_metadata_list = read_csv_as_sorted_list(self.metadata_dir)
        sorted_metadata_dict = convert_list_to_dict(sorted_metadata_list)
        sorted_processed_metadata_list = read_csv_as_sorted_list(self.metadata_processed_dir)

        while count <= self.batch_size + 1 and len(sorted_metadata_dict) > 0:
            metadata_list_row = sorted_metadata_list.pop()
            sorted_processed_metadata_list.append(metadata_list_row)
            metadata_dict_row = sorted_metadata_dict.pop()
            series = SeriesModel(self.root_directory, metadata_dict_row)
            if count == 1:
                # declare first patient in file
                patient = PatientModel(metadata_dict_row)
                patient_id_state = patient.subject_ID
                count += 1
            elif patient_id_state != series.subject_ID:
                # declare next patient
                patient = PatientModel(metadata_dict_row)
                patient_id_state = series.subject_ID

            if len(sorted_metadata_dict) == 0:
                # add last patient in file
                print("Extracted Data for " + patient.subject_ID)
                data.append(patient)
                count += 1
            elif patient_id_state != sorted_metadata_dict[len(sorted_metadata_dict) - 1]["Subject ID"]:
                # add next patient
                print("Extracted Data for " + patient.subject_ID)
                data.append(patient)
                count += 1
            patient.series.append(series)

        self.dataset = DatasetModel(self.metadata_dir, data)
        save_csv_list(self.metadata_processed_dir, sorted_processed_metadata_list)
        save_csv_list(self.metadata_dir, sorted_metadata_list)


# Initialize parser
# parser = argparse.ArgumentParser()
#
# # Adding optional argument
# parser.add_argument("-m", "--meta_data_dir", help="Absolute path to meta_data csv file")
#
# # Read arguments from command line
# args = parser.parse_args()
#
# if args.Output:
#     print("Displaying Output as: % s" % args.Output)

# if __name__ == "__main__":
#     root_directory = os.getcwd().replace("processors", "")
#     metadata_directory = "./test_data/metadata_testing.csv"
#     patient_batch_size = 2
#
#     extract_dataset = ExtractDataset(root_directory, metadata_directory, patient_batch_size)
