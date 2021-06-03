import os
import cv2
import numpy as np
from colorthief import ColorThief


def main():
    input_folder = "data/painter_by_numbers_scene_correct_color/"

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if "_dominant" in filename or "_mean" in filename:
                continue
            print(f"Working on: {filename}")
            file_path = os.path.join(root, filename)
            img = cv2.imread(file_path)

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mean_hsv_overall = np.round(np.mean(hsv_img.reshape(-1, 3), axis=0)).astype(int)
            overall_mean_img_hsv = np.tile(mean_hsv_overall, img.shape[0] * img.shape[1]).reshape(img.shape).astype(
                'uint8')
            overall_mean_img = cv2.cvtColor(overall_mean_img_hsv, cv2.COLOR_HSV2BGR)
            filename_mean = f"{file_path.split('.')[0]}_mean.{file_path.split('.')[1]}"
            cv2.imwrite(filename_mean, overall_mean_img)


            color_thief = ColorThief(file_path)
            # get the dominant color
            dominant_color = color_thief.get_color(quality=1)
            dominant_color_hsv = cv2.cvtColor(np.uint8([[list(dominant_color)]]), cv2.COLOR_RGB2HSV)
            resize_color = cv2.resize(dominant_color_hsv, (img.shape[1], img.shape[0]))
            color_img = cv2.cvtColor(resize_color, cv2.COLOR_HSV2BGR)
            filename_color = f"{file_path.split('.')[0]}_dominant.{file_path.split('.')[1]}"
            cv2.imwrite(filename_color, color_img)

            print(f"Done: {filename}")


if __name__ == '__main__':
    main()
