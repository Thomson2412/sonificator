import math
import cv2
import numpy as np

from DataStructure import DataStructure


def scan_img(input_img, steps=16):
    img = cv2.imread(input_img)
    scale_h = (1080 / 2) / img.shape[0]
    width = int(img.shape[1] * scale_h)
    height = int(img.shape[0] * scale_h)
    dim = (width, height)
    img = cv2.resize(img, dim)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edge_img = cv2.Canny(img, 100, 200)

    mean_hsv_overall = np.round(np.mean(hsv_img.reshape(-1, 3), axis=0)).astype(int)

    root = scale_between_range(mean_hsv_overall[0], 0, 179, 0, 6)
    scale = scale_between_range(mean_hsv_overall[2], 0, 179, 0, 1)

    data = DataStructure(
        root,
        scale,
        steps
    )

    step_size_x = math.ceil(width / math.sqrt(steps))
    step_size_y = math.ceil(height / math.sqrt(steps))
    for y in range(0, height, step_size_y):
        for x in range(0, width, step_size_x):
            sub_hsv = hsv_img[y:y + step_size_y, x:x + step_size_x]
            mean_hsv = np.round(np.mean(sub_hsv.reshape(-1, 3), axis=0)).astype(int)

            hue = scale_between_range(mean_hsv[0], 0, 179, 0, 6)
            saturation = scale_between_range(mean_hsv[1], 0, 255, 1, 100)
            intensity = scale_between_range(mean_hsv[2], 0, 255, 2, 5)

            sub_edge = edge_img[y:y + step_size_y, x:x + step_size_x]
            edginess = np.count_nonzero(sub_edge == 255) / sub_edge.size

            start_position = int(math.floor(len(sub_edge) / 2))
            line = []
            for i, col in enumerate(sub_edge.T):
                edge_positions = np.where(col == 255)[0]
                if len(edge_positions) > 0 and len(line) > 0:
                    close = min(edge_positions, key=lambda pos: abs(pos - line[-1]))
                    line.append(close)
                elif len(edge_positions) > 0 and len(line) == 0:
                    close = min(edge_positions, key=lambda pos: abs(pos - start_position))
                    line.append(close)
                elif len(line) > 0:
                    line.append(line[-1])
                else:
                    line.append(start_position)

            inverted_line = [len(sub_edge) - p for p in line]
            scaled_inverted_line = []
            for point in inverted_line:
                scaled = scale_between_range(point, 0, len(sub_edge), 0, 11)
                scaled_inverted_line.append(scaled)

            data.append_sub_img(
                hue,
                saturation,
                intensity,
                4,
                edginess,
                scaled_inverted_line
            )

    return data


def scale_between_range(value, in_min, in_max, out_min, out_max):
    if in_min == in_max:
        return round((out_max - out_min) / 2)
    scaled = round(out_min + (value - in_min) * ((out_max + out_min) / (in_max - in_min)))
    if scaled > out_max:
        return round(out_max)
    if scaled < out_min:
        return round(out_min)
    return scaled


if __name__ == '__main__':
    img_data = scan_img(
        "data/painter_by_numbers_scene_correct/battle_painting-Konstantin_Korovin-Polovtsian_camp_-1914-70568.jpg")
    img_data.write_to_file("test.txt")
