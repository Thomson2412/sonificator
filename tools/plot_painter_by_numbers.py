import csv
import os
import random
import re
import shutil
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def get_items_folder(input_dir, input_csv):
    filename_list = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            filename_split = filename.split("-")
            og_filename = filename_split[-1]
            filename_list.append(og_filename)

    items = []
    with open(input_csv, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for item in csv_reader:
            if item["new_filename"] in filename_list:
                items.append(item)
    return items


def plot_items(items):
    artists = [item["artist"] for item in items]
    dates = [re.sub("\D", "", item["date"].replace("c.", "").split(".")[0]) for item in items]
    genres = [item["genre"] for item in items]
    # styles = [item["style"] for item in items]
    # titles = [item["title"] for item in items]

    # plot_hist(list(reversed(sorted(artists))), (25, 80), "Artist")
    plot_hist(list(reversed(sorted(dates, key=int))), (6, 6), "Year")
    # plot_hist(list(reversed(sorted(genres))), (15, 10), "Genre")

    # plot_hist(list(reversed(sorted(artists))), (6, len(artists) / 10), "Artist", 0.4)
    # plot_hist(list(reversed(sorted(genres))), (7.5, 6), "Genre", 0.25)
    # plot_hist(list(reversed(sorted(dates, key=int))), (6, len(dates) / 10), "Year")


def plot_hist(data, size, title, left_adjust=0.0):
    unique_data_len = len(set(data))
    plt.figure(figsize=size)
    if left_adjust != 0.0:
        plt.subplots_adjust(left=left_adjust)
    n, bins, patches = plt.hist(x=data,
                                bins=np.arange(unique_data_len + 1) - 0.5,
                                color='#0504aa', alpha=0.7, rwidth=0.85,
                                orientation="horizontal")
    plt.grid(axis='x', alpha=0.75)
    plt.xlabel('Frequency')
    plt.ylabel('Data')
    plt.title(f"{title}\nTotal: {len(data)}")
    plt.xlim([0, max(n) + (max(n) * 0.1)])
    plt.ylim([-1, unique_data_len])
    for i in range(len(bins) - 1):
        plt.text(n[i] + (max(n) * 0.01), bins[i] + 0.25, str(round(n[i])))
    plt.show()
    # plt.savefig(f"plot_{title.lower()}")


if __name__ == '__main__':
    painter_by_numbers_dir = "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers/"
    painter_by_numbers_csv = os.path.join(painter_by_numbers_dir, "all_data_info.csv")
    scene_correct_dir = "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_scene_correct/"

    folder_items = get_items_folder(scene_correct_dir, painter_by_numbers_csv)
    plot_items(folder_items)
