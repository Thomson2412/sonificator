import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from pythonosc import osc_message_builder, osc_bundle_builder
from scipy.signal import savgol_filter


def scan_img():
    osc_ip = "127.0.0.1"
    osc_port = 8484
    osc_client = SimpleUDPClient(osc_ip, osc_port)

    sub_img_duration = 2000

    img = cv2.imread("data/monet/0500.png")
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    edge_img = cv2.Canny(img, 100, 200)

    cv2.imshow("HSV", hsv_img)

    step_size = 40 * 2
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            sub_img = hsv_img[y:y + step_size, x:x + step_size]
            mean_hsv = np.round(np.mean(sub_img.reshape(-1, 3), axis=0)).astype(int)

            hue = scale_between_range(mean_hsv[0], 0, 179, 0, 12)
            saturation = scale_between_range(mean_hsv[1], 0, 255, 100, 400)
            intensity = scale_between_range(mean_hsv[2], 0, 255, 1, 6)
            hsv_img[y:y + step_size, x:x + step_size] = img[y:y + step_size, x:x + step_size]

            sub_edge = edge_img[y:y + step_size, x:x + step_size]
            start_position = int(len(sub_edge) / 2)
            line = []
            for i, col in enumerate(sub_edge.T):
                edge_positions = np.where(col == 255)[0]
                if len(edge_positions) > 0:
                    close = min(edge_positions, key=lambda pos: abs(pos - start_position))
                    line.append(close)
                elif len(line) > 0:
                    line.append(line[-1])
                else:
                    line.append(start_position)

            window = len(sub_img[0]) if step_size % 2 != 0 else len(sub_img[0]) - 1
            smooth_line = np.clip(np.round(savgol_filter(line, window, 3)).astype(int), 0, len(sub_img) - 1).tolist()
            for i, point in enumerate(line):
                sub_img[point][i] = [0, 255, 0]

            cv2.imshow("HSV", hsv_img)
            cv2.imshow("Edge", edge_img)
            scaled_line = []
            for point in line:
                scaled_line.append(scale_between_range(point, min(line), max(line), 0, 100))
            # msg = create_message_from_list("/low_level_data", [hue, saturation, intensity, sub_img_duration, scaled_line])
            msg = create_message_from_list("/test", [hue, saturation, intensity, sub_img_duration, scaled_line])
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
