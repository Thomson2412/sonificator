import csv
import os
import random
import re
import shutil
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def get_items_all_csv(input_csv):
    items = []
    with open(input_csv, mode='r') as csv_file:
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


def filter_genre(items):
    block_list = {
        "caricature",
        "illustration",
        "nude painting (nu)",
        "panorama",
        "portrait",
        "poster",
        "self-portrait",
        "sketch and study",
        "still life",
        "symbolic painting",
        "vanitas"
    }

    output_items = []
    for item in items:
        if item["genre"] not in block_list:
            output_items.append(item)
    return output_items


def pick_item_per_genre(items, amount):
    output_items = []
    genres = set([item["genre"] for item in items])
    for genre in genres:
        genre_items = list(filter(lambda item: item["genre"] == genre, items))
        for item in random.sample(genre_items, min(amount, len(genre_items))):
            output_items.append(item)
    return output_items


def copy_items(input_items, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        answer = input("Remove existing dir")
        if answer.lower() == "y":
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)

    for item in input_items:
        og_filename = item["new_filename"]
        folder = "train/" if item["in_train"].lower() == "true" else "test/"
        filepath = os.path.join(f"{input_dir}{folder}", og_filename)
        shutil.copy2(filepath, output_dir)
        item_year = re.sub("\D", "", item["date"].replace("c.", "").split(".")[0])
        new_filename = "{}-{}-{}-{}-{}".format(
            "_".join(item["genre"].split(" ")),
            "_".join(item["artist"].split(" ")),
            "_".join(item["title"].split(" ")),
            item_year,
            og_filename).replace("/", "_")
        new_filepath = os.path.join(output_dir, new_filename)
        os.rename(os.path.join(output_dir, og_filename), new_filepath)


if __name__ == '__main__':
    painter_by_numbers_dir = "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers/"
    painter_by_numbers_csv = f"{painter_by_numbers_dir}all_data_info.csv"
    copy_to_dir = "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_filtered/"

    all_items = get_items_all_csv(painter_by_numbers_csv)
    impressionism = filter_items_impressionism(all_items)
    filtered_genre = filter_genre(impressionism)
    limit_item_per_genre = pick_item_per_genre(filtered_genre, 10)

    items_to_process = limit_item_per_genre
    copy_items(items_to_process, painter_by_numbers_dir, copy_to_dir)
