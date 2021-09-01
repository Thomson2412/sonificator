import cv2
import numpy as np
import Utils

CV_HSV_MIN_MAX = [(0, 179), (0, 255), (0, 255)]
ROOT_MIN_MAX = (0, 11)
SCALE_MIN_MAX = (0, 1)
KEY_STEP_MIN_MAX = (0, 6)
LOUDNESS_MIN_MAX = (1, 100)
OCTAVE_MIN_MAX = (2, 5)


def hue_to_scale(height, width, min_max):
    img = np.empty((height, width, 3), dtype=np.uint8)
    img_scaled = np.empty((height, width, 3), dtype=np.uint8)
    hsv_factor = CV_HSV_MIN_MAX[0][1] / width

    last_hue_scaled_up = 0
    for y in range(height):
        for x in range(width):
            hue = round(x * hsv_factor)
            img[y][x] = (hue, CV_HSV_MIN_MAX[1][1], CV_HSV_MIN_MAX[2][1])
            hue_scaled_down = Utils.scale_between_range(hue, CV_HSV_MIN_MAX[0], min_max)
            hue_scaled_up = Utils.scale_between_range(hue_scaled_down, min_max, CV_HSV_MIN_MAX[0])
            img_scaled[y][x] = (hue_scaled_up, CV_HSV_MIN_MAX[1][1], CV_HSV_MIN_MAX[2][1])
            if last_hue_scaled_up != hue_scaled_up:
                cv2.putText(
                    img_scaled,
                    str(hue_scaled_down),
                    (x, round(height / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    6
                )
                last_hue_scaled_up = hue_scaled_up

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_HSV2BGR)
    img = cv2.vconcat([img, img_scaled])
    # cv2.imwrite("../data/test/scale_plots/hue_6.png", img)
    cv2.imshow("data/test/scale_plots/hue_6.png", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    hue_to_scale(200, 800, KEY_STEP_MIN_MAX)
