import csv
import os
import random
import re
import shutil
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

painter_by_numbers_dir = "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers/"


def get_items():
    items = []
    with open(f"{painter_by_numbers_dir}all_data_info.csv",
              mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for item in csv_reader:
            if "" not in item.values():
                items.append(item)
    return items


def filter_items_impressionism(items):
    output_items = []
    for item in items:
        style = item["style"]
        if style == "Impressionism":
            output_items.append(item)
    return output_items


def pick_item_per_genre(items):
    block_list = {
        "caricature",
        "panorama",
        "poster",
        "self-portrait",
        "sketch and study",
        "still life",
        "symbolic painting"
    }
    output_items = []
    genres = set([item["genre"] for item in items])
    for genre in (genres - block_list):
        genre_items = list(filter(lambda item: item["genre"] == genre, items))
        for item in random.sample(genre_items, min(10, len(genre_items))):
            output_items.append(item)
    return output_items


def copy_items(input_items):
    output_folder = "data/impressionism/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)

    for item in input_items:
        og_filename = item["new_filename"]
        folder = "train/" if item["in_train"].lower() == "true" else "test/"
        filepath = os.path.join(f"{painter_by_numbers_dir}{folder}", og_filename)
        shutil.copy2(filepath, output_folder)
        item_extension = os.path.splitext(filepath)[1]
        item_year = re.sub("\D", "", item["date"].replace("c.", "").split(".")[0])
        new_filename = "{}-{}-{}-{}{}".format(
            "_".join(item["genre"].split(" ")),
            "_".join(item["artist"].split(" ")),
            "_".join(item["title"].split(" ")),
            item_year,
            item_extension)
        new_filepath = os.path.join(output_folder, new_filename)
        os.rename(os.path.join(output_folder, og_filename), new_filepath)


def plot_items(items):
    artists = [item["artist"] for item in items]
    dates = [re.sub("\D", "", item["date"].replace("c.", "").split(".")[0]) for item in items]
    # genres = [item["genre"] for item in items]
    # styles = [item["style"] for item in items]
    # titles = [item["title"] for item in items]

    # print_counted(artists)
    # print_counted(dates)
    # print_counted(genres)
    # print_counted(styles)
    # print_counted(titles)

    # plot_hist(list(reversed(sorted(artists))), (25, 80), "Artist")
    # plot_hist(list(reversed(sorted(dates, key=int))), (6, 40), "Year")
    # plot_hist(list(reversed(sorted(genres))), (15, 10), "Genre")

    plot_hist(list(reversed(sorted(artists))), (6, len(artists) / 10), "Artist", 0.4)
    plot_hist(list(reversed(sorted(dates, key=int))), (6, len(dates) / 10), "Year")


def print_counted(input_list):
    counted = Counter(input_list)
    for item, count in counted.items():
        print(f"{item}: {count}")


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
    plt.title(title)
    plt.xlim([0, max(n) + (max(n) * 0.1)])
    plt.ylim([-1, unique_data_len])
    for i in range(len(bins) - 1):
        plt.text(n[i] + (max(n) * 0.01), bins[i] + 0.25, str(round(n[i])))
    # plt.show()
    plt.savefig(f"plot_{title.lower()}")


if __name__ == '__main__':
    items_to_process = pick_item_per_genre(filter_items_impressionism(get_items()))
    plot_items(items_to_process)
    copy_items(items_to_process)
