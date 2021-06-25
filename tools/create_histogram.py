import os
import cv2
import numpy as np
from colorthief import ColorThief
from matplotlib import pyplot as plt


def main():
    # painter_by_numbers_dir = "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_scene_correct/"
    input_dir = "/mnt/datadrive/projects/thesis/sonificator/data/test/"
    bins = 256

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if ".png" in filename or ".jpg" in filename:
                file_path = os.path.join(root, filename)
                img = cv2.imread(file_path)

                hist = cv2.calcHist([img], [0], None, [bins], [0, bins])
                plt.figure()
                plt.title(f"{filename} Grayscale Histogram")
                plt.xlabel("Bins")
                plt.ylabel("% of Pixels")
                plt.plot(hist)
                plt.xlim([0, bins])
                plt.show()

                points = [str(item[0]) for item in hist]
                filename_hist = os.path.join(root, f"{os.path.splitext(filename)[0]}_hist.txt")
                with open(filename_hist, "w") as f:
                    f.write(" ".join(points))
                print(f"Done: {filename}")


if __name__ == '__main__':
    main()
