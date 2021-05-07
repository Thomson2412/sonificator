import csv
import math
import os
import re
import shutil
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def get_items():
    items = []
    with open('/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers/all_data_info.csv',
              mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for item in csv_reader:
            if "" not in item.values():
                items.append(item)
    return items


def filter_items(input_items):
    output_items = []
    for item in input_items:
        style = item["style"]
        if style == "Impressionism":
            output_items.append(item)
    return output_items


def copy_items():
    file_list = []
    for item in filter_items(get_items()):
        file_list.append(item["new_filename"])

    folders = [
        "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers/train/",
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


def plot_items():
    items = filter_items(get_items())

    artists = [item["artist"] for item in items]
    dates = [re.sub("\D", "", item["date"].replace("c.", "").split(".")[0]) for item in items]
    genres = [item["genre"] for item in items]
    styles = [item["style"] for item in items]
    titles = [item["title"] for item in items]

    # print_counted(artists)
    # print_counted(dates)
    # print_counted(genres)
    # print_counted(styles)
    # print_counted(titles)

    plot_hist(list(reversed(sorted(artists))), (25, 80), "Artist")
    plot_hist(list(reversed(sorted(dates, key=int))), (6, 40), "Year")
    plot_hist(list(reversed(sorted(genres))), (15, 10), "Genre")


def print_counted(input_list):
    counted = Counter(input_list)
    for item, count in counted.items():
        print(f"{item}: {count}")


def plot_hist(data, size, title):
    unique_data_len = len(set(data))
    plt.figure(figsize=size)
    n, bins, patches = plt.hist(x=data,
                                bins=np.arange(unique_data_len + 1) - 0.5,
                                color='#0504aa', alpha=0.7, rwidth=0.85,
                                orientation="horizontal")
    plt.grid(axis='x', alpha=0.75)
    plt.xlabel('Frequency')
    plt.ylabel('Data')
    plt.title(title)
    # plt.xscale("log")
    plt.xlim([0, max(n) + (max(n) * 0.1)])
    plt.ylim([-1, unique_data_len])
    for i in range(len(bins) - 1):
        plt.text(n[i] + (max(n) * 0.01), bins[i] + 0.25, str(round(n[i])))

    plt.show()


if __name__ == '__main__':
    # copy_items()
    plot_items()