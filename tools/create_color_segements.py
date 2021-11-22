import math
import os
import cv2
import numpy as np
import Utils

CV_HSV_MIN_MAX = [(0, 179), (0, 255), (0, 255)]

ROOT_MIN_MAX = (0, 11)
SCALE_MIN_MAX = (0, 1)
KEY_STEP_MIN_MAX = (0, 6)
LOUDNESS_MIN_MAX = (1, 100)
OCTAVE_MIN_MAX = (2, 5)
PAN_MIN_MAX = (0, 1500)
MELODY_NOTE_AMOUNT = 4
DURATION = 4


def main(output_individual=False):
    input_folder = "../data/painter_by_numbers_scene_correct_segments_dominant/"
    steps = 16
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            individual_segments = []
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
            presentation_img = np.array(img, copy=True)

            step_size_x = math.ceil(width / math.sqrt(steps))
            step_size_y = math.ceil(height / math.sqrt(steps))

            current_step = 0

            for y in range(0, height, step_size_y):
                for x in range(0, width, step_size_x):
                    sub_img = img[y:y + step_size_y, x:x + step_size_x]
                    sub_img_reshape = sub_img.reshape(-1, 3)
                    dominant_color = Utils.get_dominant_color(sub_img_reshape, 1)
                    dominant_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV).flatten()

                    presentation = np.tile(dominant_hsv, sub_img.shape[1] * sub_img.shape[0])
                    presentation = presentation.reshape(sub_img.shape).astype('uint8')
                    presentation = cv2.cvtColor(presentation, cv2.COLOR_HSV2BGR)
                    presentation_img[y:y + step_size_y, x:x + step_size_x] = presentation

                    if output_individual:
                        color_segment_img = np.full(img.shape, 255)
                        color_segment_img[y:y + step_size_y, x:x + step_size_x] = \
                            cv2.cvtColor(np.uint8([[dominant_hsv]]), cv2.COLOR_HSV2BGR)
                        filename_segment = os.path.join(
                            root,
                            f"{os.path.splitext(filename)[0]}_segment_{current_step}{os.path.splitext(filename)[1]}")
                        individual_segments.append({
                            "filename": filename_segment,
                            "img": color_segment_img
                        })

                    current_step += 1

            filename_segments = os.path.join(root,
                                             f"{os.path.splitext(filename)[0]}_segments{os.path.splitext(filename)[1]}")
            cv2.imwrite(filename_segments, presentation_img)

            if output_individual:
                saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
                (success, saliency_map) = saliency.computeSaliency(img)
                saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                priority_list = Utils.calculate_step_priority_standard(saliency_map, steps)
                for i, key in enumerate(priority_list):
                    cv2.imwrite(individual_segments[key]["filename"], individual_segments[i]["img"])

            print(f"Done: {filename}")


if __name__ == '__main__':
    main(True)
