import random
from collections import Counter
import cv2
import ObjectDetectionVisual
import SceneDetectionAudio
import Utils
from DataStructureAudio import DataStructureAudio
from DataStructureVisual import DataStructureVisual
from colorthief import ColorThief
import numpy as np
import math
import scipy
import scipy.misc
import scipy.cluster

CV_HSV_MIN_MAX = [(0, 179), (0, 255), (0, 255)]

ROOT_MIN_MAX = (0, 11)
SCALE_MIN_MAX = (0, 1)
KEY_STEP_MIN_MAX = (0, 6)
LOUDNESS_MIN_MAX = (1, 100)
OCTAVE_MIN_MAX = (2, 5)
PAN_MIN_MAX = (0, 1500)
MELODY_NOTE_AMOUNT = 4
DURATION = 4


def scan_img(input_img, steps, saliency, use_saliency, scene_detection, use_object_nav):
    img = cv2.imread(input_img)
    scale_h = (1080 / 2) / img.shape[0]
    width = int(img.shape[1] * scale_h)
    height = int(img.shape[0] * scale_h)
    dim = (width, height)
    img = cv2.resize(img, dim)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edge_img = cv2.Canny(img, 100, 200)

    (success, saliency_map) = saliency.computeSaliency(img)
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    thresh_map = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    color_thief = ColorThief(input_img)
    dominant_color = color_thief.get_color(quality=1)
    dominant_color_hsv = cv2.cvtColor(np.uint8([[list(dominant_color)]]), cv2.COLOR_RGB2HSV).flatten()
    dominant_color_img = np.tile(dominant_color_hsv, img.shape[1] * img.shape[0])
    dominant_color_img = dominant_color_img.reshape(img.shape).astype('uint8')
    dominant_color_img = cv2.cvtColor(dominant_color_img, cv2.COLOR_HSV2BGR)

    # mean_hsv_overall = np.round(np.mean(hsv_img.reshape(-1, 3), axis=0)).astype(int)

    root = Utils.scale_between_range(dominant_color_hsv[0], CV_HSV_MIN_MAX[0], ROOT_MIN_MAX)
    scale = Utils.scale_between_range(dominant_color_hsv[2], CV_HSV_MIN_MAX[2], SCALE_MIN_MAX)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_rev_flip = np.flipud(hist * -1)
    wave = np.append(hist, hist_rev_flip, 0)
    wave_str = " ".join([str(int(item[0])) for item in wave])

    scene_audio_path = " "
    if scene_detection:
        scene = scene_detection[0].detect(input_img)
        scene_audio_paths = scene_detection[1].get_audio_for_scene_folder(
            "/mnt/datadrive/projects/thesis/Datasets/Audio/Audio_Filtered", scene)
        if len(scene_audio_paths) > 0:
            scene_audio_path = random.choice(scene_audio_paths)
            print(f"{scene}: {scene_audio_path}")
        else:
            return None

    if use_object_nav:
        segmentation_img, segmentation_info = ObjectDetectionVisual.detect_panoptic(img)
        Utils.merge_segments_by_category(segmentation_img, segmentation_info)
        if use_saliency:
            priority_list = Utils.calculate_step_priority_object(segmentation_img, thresh_map)
        else:
            counts = Counter(segmentation_img.flatten())
            priority_list = list(
                {k: v for k, v in sorted(counts.items(), reverse=False, key=lambda item: item[1])}.keys())

        data_visual = DataStructureVisual(
            img,
            hsv_img,
            edge_img,
            dominant_color_img,
            len(priority_list)
        )

        data_audio = DataStructureAudio(
            root,
            scale,
            MELODY_NOTE_AMOUNT,
            wave_str,
            scene_audio_path,
            len(priority_list)
        )

        return scan_img_seg_object(segmentation_img, img, edge_img, data_visual, data_audio, priority_list)
    else:
        if use_saliency:
            priority_list = Utils.calculate_step_priority_standard(saliency_map, steps)
        else:
            priority_list = list(range(steps))
        data_visual = DataStructureVisual(
            img,
            hsv_img,
            edge_img,
            dominant_color_img,
            steps
        )

        data_audio = DataStructureAudio(
            root,
            scale,
            MELODY_NOTE_AMOUNT,
            wave_str,
            scene_audio_path,
            steps
        )
        return scan_img_seg_standard(width, height, steps, img, hsv_img, edge_img, saliency_map, thresh_map,
                                     data_visual, data_audio, use_saliency, priority_list)


def scan_img_seg_standard(width, height, steps, img, hsv_img, edge_img, saliency_map, thresh_map,
                          data_visual, data_audio, use_saliency, priority_list):
    current_step = 0
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

            if sum_sub_sal > 127 and use_saliency:
                to_use_hsv = []
                for i, item in enumerate(sub_thresh_reshape):
                    if item == 255:
                        to_use_hsv.append(sub_hsv_reshape[i])
                mean_hsv = np.round(np.mean(to_use_hsv, axis=0)).astype(int)
            else:
                mean_hsv = np.round(np.mean(sub_hsv_reshape, axis=0)).astype(int)

            hue = Utils.scale_between_range(mean_hsv[0], CV_HSV_MIN_MAX[0], KEY_STEP_MIN_MAX)
            saturation = Utils.scale_between_range(mean_hsv[1], CV_HSV_MIN_MAX[1], LOUDNESS_MIN_MAX)
            intensity = Utils.scale_between_range(mean_hsv[2], CV_HSV_MIN_MAX[2], OCTAVE_MIN_MAX)

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

            presentation = np.tile(mean_hsv, sub_hsv.shape[1] * sub_hsv.shape[0])
            presentation = presentation.reshape(sub_hsv.shape).astype('uint8')
            presentation = cv2.cvtColor(presentation, cv2.COLOR_HSV2BGR)
            inverted_color = cv2.bitwise_not(presentation.reshape(-1, 3)[0]).flatten()
            for i, point in enumerate(line):
                presentation[point][i] = inverted_color

            presentation_img = np.array(img, copy=True)
            presentation_img[y:y + step_size_y, x:x + step_size_x] = presentation
            data_visual.append_sub_img(
                presentation_img,
                DURATION,
                priority_list[current_step]
            )

            pan = int((PAN_MIN_MAX[1] / -2) + Utils.scale_between_range(
                current_step % math.sqrt(steps),
                (0, math.sqrt(steps) - 1),
                PAN_MIN_MAX))

            inverted_line = [len(sub_edge) - p for p in line]
            scaled_inverted_line = []
            for point in inverted_line:
                scaled = Utils.scale_between_range(point, (0, len(sub_edge)), (0, 11))
                scaled_inverted_line.append(scaled)

            data_audio.append_sub_img(
                hue,
                saturation,
                intensity,
                pan,
                DURATION,
                edginess,
                scaled_inverted_line,
                priority_list[current_step]
            )

            current_step += 1

    return data_audio, data_visual


def scan_img_seg_object(segmentation_img, img, edge_img, data_visual, data_audio, priority_list):
    total_pixel_count = img.shape[0] * img.shape[1]
    for current_step, mask_id in enumerate(priority_list):
        mask = np.array(segmentation_img == mask_id)
        sub_img_reshape = img[mask]

        dominant_color = get_dominant_color(sub_img_reshape, 1)
        dominant_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV).flatten()


        hue = Utils.scale_between_range(dominant_hsv[0], CV_HSV_MIN_MAX[0], KEY_STEP_MIN_MAX)
        saturation = Utils.scale_between_range(dominant_hsv[1], CV_HSV_MIN_MAX[1], LOUDNESS_MIN_MAX)
        intensity = Utils.scale_between_range(dominant_hsv[2], CV_HSV_MIN_MAX[2], OCTAVE_MIN_MAX)

        sub_edge_reshape = edge_img[mask]
        edginess = np.count_nonzero(sub_edge_reshape == 255) / sub_edge_reshape.size

        segment_pixel_count = sub_img_reshape.shape[0]
        area_percentage = (segment_pixel_count / total_pixel_count) * 100
        duration = Utils.scale_between_range(area_percentage, (0, 100), (2, 8))

        presentation_img = np.array(img, copy=True)
        presentation_img[mask] = cv2.cvtColor(np.uint8([[dominant_hsv]]), cv2.COLOR_HSV2BGR)
        data_visual.append_sub_img(
            presentation_img,
            duration,
            current_step
        )

        middle_x = round(mask.shape[1] / 2)
        left_percentage = Counter(mask.T[0:middle_x].flatten())[True] / segment_pixel_count
        right_percentage = Counter(mask.T[middle_x:img.shape[1]].flatten())[True] / segment_pixel_count
        pan = (-1 * left_percentage) + (1 * right_percentage)

        melody_array = [0, 2, 4, 6, 4, 6, 11, 2, 4, 8, 11, 8, 0]
        random.shuffle(melody_array)

        data_audio.append_sub_img(
            hue,
            saturation,
            intensity,
            pan,
            duration,
            edginess,
            melody_array,
            current_step
        )

    return data_audio, data_visual


def get_dominant_color(ar, num_clusters):
    ar = ar.astype(float)
    codes, _ = scipy.cluster.vq.kmeans2(ar, num_clusters)
    vecs, _ = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, _ = scipy.histogram(vecs, len(codes))  # count occurrences
    index_max = scipy.argmax(counts)  # find most frequent
    peak = codes[index_max]
    return peak.astype(np.uint8)


class ImageScanner:
    pass
