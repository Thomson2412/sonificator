import csv
import os
import shutil

file_list = []

with open('/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers/all_data_info.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for item in csv_reader:
        style = item["style"]
        if style == "Impressionism":
            file_list.append(item["new_filename"])

folders = ["/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers/train/",
           "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers/test/"
           ]
output_folder = "data/impressionism/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for folder in folders:
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename in file_list:
                shutil.copy2(os.path.join(root, filename), output_folder)