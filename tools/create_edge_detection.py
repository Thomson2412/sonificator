import os
import cv2
import numpy as np
from colorthief import ColorThief
from matplotlib import pyplot as plt


def main():
    input_folder = "/mnt/datadrive/projects/thesis/sonificator/data/painter_by_numbers_scene_correct_edge"

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if "_edge" in filename:
                continue
            print(f"Working on: {filename}")
            file_path = os.path.join(root, filename)
            img = cv2.imread(file_path)
            edge_img = cv2.Canny(img, 100, 200)
            filename_mean = f"{file_path.split('.')[0]}_edge.{file_path.split('.')[1]}"
            cv2.imwrite(filename_mean, edge_img)

            print(f"Done: {filename}")


if __name__ == '__main__':
    main()
