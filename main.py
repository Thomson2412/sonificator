import math
import os
import re
import cv2
import numpy as np
import subprocess
import Utils
from DataStructureAudio import DataStructureAudio
from DataStructureVisual import DataStructureVisual
from colorthief import ColorThief

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

CV_HSV_MIN_MAX = [(0, 179), (0, 255), (0, 255)]

ROOT_MIN_MAX = (0, 6)
SCALE_MIN_MAX = (0, 1)
KEY_STEP_MIN_MAX = (0, 6)
LOUDNESS_MIN_MAX = (0, 6)
OCTAVE_MIN_MAX = (2, 5)
DURATION = 4


def scan_img(input_img, steps, saliency, use_saliency):
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
    saliency_heatmap_img = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    saliency_heatmap_img = cv2.addWeighted(img, 0.3, saliency_heatmap_img, 0.7, 0)

    if use_saliency:
        priority_list = Utils.calculate_step_priority(saliency_map, steps)
    else:
        priority_list = list(range(steps))

    color_thief = ColorThief(input_img)
    dominant_color = color_thief.get_color(quality=1)
    dominant_color_hsv = cv2.cvtColor(np.uint8([[list(dominant_color)]]), cv2.COLOR_RGB2HSV).flatten()
    dominant_color_img = np.tile(dominant_color_hsv, img.shape[1] * img.shape[0])
    dominant_color_img = dominant_color_img.reshape(img.shape).astype('uint8')
    dominant_color_img = cv2.cvtColor(dominant_color_img, cv2.COLOR_HSV2BGR)

    # mean_hsv_overall = np.round(np.mean(hsv_img.reshape(-1, 3), axis=0)).astype(int)

    data_visual = DataStructureVisual(
        img,
        hsv_img,
        edge_img,
        dominant_color_img,
        saliency_heatmap_img,
        thresh_map,
        steps
    )

    root = Utils.scale_between_range(dominant_color_hsv[0], CV_HSV_MIN_MAX[0], ROOT_MIN_MAX)
    scale = Utils.scale_between_range(dominant_color_hsv[2], CV_HSV_MIN_MAX[0], SCALE_MIN_MAX)

    data_audio = DataStructureAudio(
        root,
        scale,
        steps
    )

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

            data_visual.append_sub_img(
                presentation,
                x,
                y,
                DURATION,
                priority_list[current_step]
            )

            inverted_line = [len(sub_edge) - p for p in line]
            scaled_inverted_line = []
            for point in inverted_line:
                scaled = Utils.scale_between_range(point, (0, len(sub_edge)), (0, 11))
                scaled_inverted_line.append(scaled)

            data_audio.append_sub_img(
                hue,
                saturation,
                intensity,
                DURATION,
                edginess,
                scaled_inverted_line,
                priority_list[current_step]
            )

            current_step += 1

    return data_audio, data_visual


def convert_paintings_to_txt(input_dir, output_dir, with_saliency=False):
    saliency_fine = cv2.saliency.StaticSaliencyFineGrained_create()
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if ".jpg" in filename or ".png" in filename:
                file_path = os.path.join(root, filename)
                audio_data = scan_img(file_path, 16, saliency_fine, with_saliency)[0]
                output_file = os.path.join(output_dir, f"{filename.split('.')[0]}.txt")
                audio_data.write_to_file(output_file)


def convert_txt_to_sound(exec_file, input_dir):
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            print(f"Begin: {filename}")
            exec_path = os.path.abspath(exec_file)
            input_file_path = os.path.abspath(os.path.join(root, filename))
            output_filepath = os.path.abspath(os.path.join(root, f"{filename.split('.')[0]}.aiff"))
            p = subprocess.Popen([
                "sclang",
                exec_path,
                input_file_path,
                output_filepath
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (output, err) = p.communicate()
            p_status = p.wait()
            if p_status == 0:
                r = re.findall('(ERROR)+', str(output), flags=re.IGNORECASE)
                if r:
                    print('An ERROR occurred!')
                else:
                    print('File is executable!')
            print("End")


def convert_paintings_to_presentation(input_dir, output_dir, with_saliency=False):
    saliency_fine = cv2.saliency.StaticSaliencyFineGrained_create()
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if ".jpg" in filename or ".png" in filename:
                file_path = os.path.join(root, filename)
                visual_data = scan_img(file_path, 16, saliency_fine, with_saliency)[1]
                # for i in range(16):
                #     presentation = visual_data.get_presentation_for_step(i, False)
                #     output_file = os.path.join(output_dir, f"{filename.split('.')[0]}_presentation_{i}.png")
                #     cv2.imwrite(output_file, presentation)
                #
                #     segment = visual_data.get_segment_for_step(i, False)
                #     output_file = os.path.join(output_dir, f"{filename.split('.')[0]}_segment_{i}.png")
                #     cv2.imwrite(output_file, segment)
                output_file = os.path.join(output_dir, f"{filename.split('.')[0]}_vid.avi")
                visual_data.generate_presentation_video(output_file, True)


if __name__ == '__main__':
    # convert_paintings_to_txt("data/painter_by_numbers_scene_correct/", "converted/")
    # convert_txt_to_sound("sound_engine.scd", "converted/")
    # convert_paintings_to_txt("data/test/", "data/test/", True)
    convert_paintings_to_presentation("data/test/", "data/presentation/", False)
    convert_paintings_to_presentation("data/test/", "data/presentation_saliency/", True)
