import math

import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from pythonosc import osc_message_builder, osc_bundle_builder
from scipy.signal import savgol_filter


def scan_img():
    osc_ip = "127.0.0.1"
    osc_port = 8484
    osc_client = SimpleUDPClient(osc_ip, osc_port)

    sub_img_duration = 4000

    # img = cv2.imread("data/monet/0000.png")
    img = cv2.imread("data/bob_ross/painting10.png")
    scale_h = (1080 / 2) / img.shape[0]
    width = int(img.shape[1] * scale_h)
    height = int(img.shape[0] * scale_h)
    dim = (width, height)
    img = cv2.resize(img, dim)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    edge_img = cv2.Canny(img, 100, 200)

    cv2.imshow("HSV", hsv_img)

    steps = 8
    step_size_x = math.ceil(width / steps)
    step_size_y = math.ceil(height / steps)
    for y in range(0, height, step_size_y):
        for x in range(0, width, step_size_x):
            sub_img = hsv_img[y:y + step_size_y, x:x + step_size_x]
            mean_hsv = np.round(np.mean(sub_img.reshape(-1, 3), axis=0)).astype(int)

            hue = scale_between_range(mean_hsv[0], 0, 179, 0, 12)
            saturation = scale_between_range(mean_hsv[1], 0, 255, 100, 400)
            intensity = scale_between_range(mean_hsv[2], 0, 255, 1, 6)
            hsv_img[y:y + step_size_y, x:x + step_size_x] = img[y:y + step_size_y, x:x + step_size_x]

            sub_edge = edge_img[y:y + step_size_y, x:x + step_size_x]
            edginess = np.count_nonzero(sub_edge == 255) / sub_edge.size

            start_position = int(len(sub_edge) / 2)
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

            #window = len(sub_img[0]) if step_size % 2 != 0 else len(sub_img[0]) - 1
            #smooth_line = np.clip(np.round(savgol_filter(line, window, 3)).astype(int), 0, len(sub_img) - 1).tolist()
            for i, point in enumerate(line):
                sub_img[point][i] = [0, 255, 0]

            cv2.imshow("HSV", hsv_img)
            cv2.imshow("Edge", edge_img)

            scaled_line = []
            for point in line:
                scaled_line.append(scale_between_range(point, min(line), max(line), 0, 11))

            msg = create_message_from_list("/low_level_data", [
                hue,
                saturation,
                intensity,
                sub_img_duration,
                edginess,
                line])
            # msg = create_message_from_list("/test", [hue, saturation, intensity, sub_img_duration, scaled_line])
            osc_client.send(msg)

            cv2.waitKey(sub_img_duration)
    cv2.destroyAllWindows()


def create_message_from_list(address, data_list):
    msg = osc_message_builder.OscMessageBuilder(address=address)
    for data in data_list:
        if isinstance(data, list):
            msg.add_arg(bytes(data))
        else:
            msg.add_arg(data)
    return msg.build()


def scale_between_range(value, in_min, in_max, out_min, out_max):
    if in_min == in_max:
        return value
    scaled = round(out_min + (value - in_min) * ((out_max + out_min) / (in_max - in_min)))
    if scaled > out_max:
        return out_max
    if scaled < out_min:
        return out_min
    return scaled


if __name__ == '__main__':
    scan_img()
