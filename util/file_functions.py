import csv
import operator


def read_csv_as_sorted_list(directory):
    try:
        reader = csv.reader(open(directory))
        return sorted(reader, key=operator.itemgetter(4), reverse=True)
    except FileNotFoundError:
        print(directory+" file not found")
        return []

def read_txt_and_append_to_list(directory, read_list):
    try:
        with open(directory, 'r') as csvfile:
            feature_reader = csv.reader(csvfile)
            for directory in feature_reader:
                read_list.append(directory[0])
    except FileNotFoundError:
        print(directory+" file not found")

def save_csv_list(directory, save_list):
    with open(directory, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for line in save_list:
            writer.writerow(line)
