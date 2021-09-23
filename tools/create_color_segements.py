import math
import os
import cv2
import numpy as np
import Utils


def main():
    input_folder = "../data/painter_by_numbers_scene_correct_segments_dominant/"
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    steps = 16
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if "_segments" in filename:
                continue
            print(f"Working on: {filename}")
            file_path = os.path.abspath(os.path.join(root, filename))

            img = cv2.imread(file_path)
            if img is None:
                continue
            scale_h = (1080 / 2) / img.shape[0]
            width = int(img.shape[1] * scale_h)
            height = int(img.shape[0] * scale_h)
            dim = (width, height)
            img = cv2.resize(img, dim)

            step_size_x = math.ceil(width / math.sqrt(steps))
            step_size_y = math.ceil(height / math.sqrt(steps))
            for y in range(0, height, step_size_y):
                for x in range(0, width, step_size_x):
                    sub_img = img[y:y + step_size_y, x:x + step_size_x]
                    sub_img_reshape = sub_img.reshape((sub_img.shape[1] * sub_img.shape[0], 3))

                    dominant_color = Utils.get_dominant_color(sub_img_reshape, 1)
                    dominant_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV).flatten()

                    presentation = np.tile(dominant_hsv, sub_img.shape[1] * sub_img.shape[0])
                    presentation = presentation.reshape(sub_img.shape).astype('uint8')
                    presentation = cv2.cvtColor(presentation, cv2.COLOR_HSV2BGR)
                    img[y:y + step_size_y, x:x + step_size_x] = presentation

            filename_segments = os.path.join(root,
                                             f"{os.path.splitext(filename)[0]}_segments{os.path.splitext(filename)[1]}")
            cv2.imwrite(filename_segments, img)

            print(f"Done: {filename}")


if __name__ == '__main__':
    main()
