import math
import os
import cv2
import numpy as np


def main():
    input_folder = "/mnt/datadrive/projects/thesis/sonificator/data/painter_by_numbers_scene_correct_segments/"
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    steps = 16
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if "_segments" in filename:
                continue
            print(f"Working on: {filename}")
            file_path = os.path.join(root, filename)

            img = cv2.imread(file_path)
            scale_h = (1080 / 2) / img.shape[0]
            width = int(img.shape[1] * scale_h)
            height = int(img.shape[0] * scale_h)
            dim = (width, height)
            img = cv2.resize(img, dim)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            (success, saliency_map) = saliency.computeSaliency(img)
            saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            thresh_map = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            step_size_x = math.ceil(width / math.sqrt(steps))
            step_size_y = math.ceil(height / math.sqrt(steps))
            for y in range(0, height, step_size_y):
                for x in range(0, width, step_size_x):
                    sub_sal = saliency_map[y:y + step_size_y, x:x + step_size_x]
                    sub_thresh = thresh_map[y:y + step_size_y, x:x + step_size_x]
                    sum_sub_sal = np.round(np.sum(sub_sal) / (sub_sal.shape[1] * sub_sal.shape[0])).astype(int)

                    sub_hsv = hsv_img[y:y + step_size_y, x:x + step_size_x]
                    sub_hsv_reshape = sub_hsv.reshape(-1, 3)
                    sub_thresh_reshape = sub_thresh.flatten()

                    if sum_sub_sal > 127:
                        to_use_hsv = []
                        for i, item in enumerate(sub_thresh_reshape):
                            if item == 255:
                                to_use_hsv.append(sub_hsv_reshape[i])
                        mean_hsv = np.round(np.mean(to_use_hsv, axis=0)).astype(int)
                    else:
                        mean_hsv = np.round(np.mean(sub_hsv_reshape, axis=0)).astype(int)

                    presentation = np.tile(mean_hsv, sub_hsv.shape[1] * sub_hsv.shape[0])
                    presentation = presentation.reshape(sub_hsv.shape).astype('uint8')
                    presentation = cv2.cvtColor(presentation, cv2.COLOR_HSV2BGR)
                    img[y:y + step_size_y, x:x + step_size_x] = presentation

            filename_segments = os.path.join(root,
                                             f"{os.path.splitext(filename)[0]}_segments{os.path.splitext(filename)[1]}")
            cv2.imwrite(filename_segments, img)

            print(f"Done: {filename}")


if __name__ == '__main__':
    main()
