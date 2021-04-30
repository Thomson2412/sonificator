import math
import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from pythonosc import osc_message_builder, osc_bundle_builder
# from scipy.signal import savgol_filter


def scan_img():
    osc_ip = "127.0.0.1"
    osc_port = 8484
    osc_client = SimpleUDPClient(osc_ip, osc_port)

    sub_img_duration = 4000

    # img = cv2.imread("data/monet/0000.png")
    # img = cv2.imread("data/monet/0010.png")
    img = cv2.imread("data/monet/0015.png")
    # img = cv2.imread("data/monet/0018.png")
    # img = cv2.imread("data/monet/0020.png")
    # img = cv2.imread("data/monet/0027.png")
    # img = cv2.imread("data/monet/0032.png")
    # img = cv2.imread("data/monet/0073.png")
    # img = cv2.imread("data/bob_ross/painting10.png")
    # img = cv2.imread("data/bob_ross/painting16.png")
    # img = cv2.imread("data/shu/Jackson_Pollock.png")
    # img = cv2.imread("data/shu/Mark_Rothko.png")
    scale_h = (1080 / 2) / img.shape[0]
    width = int(img.shape[1] * scale_h)
    height = int(img.shape[0] * scale_h)
    dim = (width, height)
    img = cv2.resize(img, dim)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edge_img = cv2.Canny(img, 100, 200)
    presentation = np.array(img, copy=True)

    cv2.imshow("OG", img)
    cv2.imshow("HSV", hsv_img)
    cv2.imshow("Edge", edge_img)
    cv2.imshow("Presentation", presentation)

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    cv2.imshow("Output", saliencyMap)
    cv2.waitKey(0)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    # if we would like a *binary* map that we could process for contours,
    # compute convex hull's, extract bounding boxes, etc., we can
    # additionally threshold the saliency map
    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # show the images
    cv2.imshow("Output", saliencyMap)
    cv2.imshow("Thresh", threshMap)
    cv2.waitKey(0)

    saliency = cv2.saliency.ObjectnessBING_create()
    saliency.setTrainingPath("model")

    # compute the bounding box predictions used to indicate saliency
    (success, saliencyMap) = saliency.computeSaliency(img)
    numDetections = saliencyMap.shape[0]
    # loop over the detections
    for i in range(0, min(numDetections, 6)):
        # extract the bounding box coordinates
        (startX, startY, endX, endY) = saliencyMap[i].flatten()

        # randomly generate a color for the object and draw it on the image
        output = img.copy()
        color = np.random.randint(0, 255, size=(3,))
        color = [int(c) for c in color]
        cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
        # show the output image
        cv2.imshow("Image", output)
        cv2.waitKey(0)

    mean_hsv_overall = np.round(np.mean(hsv_img.reshape(-1, 3), axis=0)).astype(int)
    overall_mean_img_hsv = np.tile(mean_hsv_overall, img.shape[0] * img.shape[1]).reshape(img.shape).astype('uint8')
    overall_mean_img = cv2.cvtColor(overall_mean_img_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("Overall", overall_mean_img)
    root = scale_between_range(mean_hsv_overall[0], 0, 179, 0, 6)
    scale = scale_between_range(mean_hsv_overall[2], 0, 179, 0, 1)

    steps = 16
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

            # window = len(sub_img[0]) if step_size % 2 != 0 else len(sub_img[0]) - 1
            # smooth_line = np.clip(np.round(savgol_filter(line, window, 3)).astype(int), 0, len(sub_img) - 1).tolist()

            sub_mean = np.tile(mean_hsv, sub_hsv.shape[0] * sub_hsv.shape[1])
            sub_mean = sub_mean.reshape(sub_hsv.shape).astype('uint8')
            sub_mean = cv2.cvtColor(sub_mean, cv2.COLOR_HSV2BGR)
            inverted_color = cv2.bitwise_not(sub_mean.reshape(-1, 3)[0]).flatten()
            sub_img = img[y:y + step_size_y, x:x + step_size_x]
            for i, point in enumerate(line):
                # sub_mean[point][i] = inverted_color
                sub_img[point][i] = [0,0,255]# inverted_color
            presentation[y:y + step_size_y, x:x + step_size_x] = sub_mean

            cv2.imshow("Presentation", presentation)
            cv2.imshow("OG", img)

            inverted_line = [len(sub_edge) - p for p in line]
            scaled_inverted_line = []
            for point in inverted_line:
                scaled = scale_between_range(point, 0, len(sub_edge), 0, 11)
                scaled_inverted_line.append(scaled)

            msg = create_message_from_list("/low_level_data", [
                hue,
                saturation,
                intensity,
                sub_img_duration,
                edginess,
                scaled_inverted_line,
                root,
                scale])
            osc_client.send(msg)

            # cv2.waitKey(sub_img_duration)
            cv2.waitKey(1)
    # cv2.imwrite("Segments.png", presentation)
    # cv2.imwrite("Edge.png", edge_img)
    cv2.imwrite("overall_mean.png", overall_mean_img)
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
        return round((out_max - out_min) / 2)
    scaled = round(out_min + (value - in_min) * ((out_max + out_min) / (in_max - in_min)))
    if scaled > out_max:
        return out_max
    if scaled < out_min:
        return out_min
    return scaled


if __name__ == '__main__':
    scan_img()
