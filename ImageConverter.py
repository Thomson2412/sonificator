import math
import os
import re
import cv2
import numpy as np
import subprocess

import SceneDetectionAudio
import Utils
from DataStructureAudio import DataStructureAudio
from DataStructureVisual import DataStructureVisual
from colorthief import ColorThief
from SceneDetectionVisual import SceneDetectionVisual

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

STEPS = 16

CV_HSV_MIN_MAX = [(0, 179), (0, 255), (0, 255)]

ROOT_MIN_MAX = (0, 6)
SCALE_MIN_MAX = (0, 1)
KEY_STEP_MIN_MAX = (0, 6)
LOUDNESS_MIN_MAX = (1, 100)
OCTAVE_MIN_MAX = (2, 5)
PAN_MIN_MAX = (0, 1500)
DURATION = 4


def scan_img(input_img, steps, saliency, use_saliency, scene_detection, use_scene):
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
    scale = Utils.scale_between_range(dominant_color_hsv[2], CV_HSV_MIN_MAX[2], SCALE_MIN_MAX)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_rev_flip = np.flipud(hist * -1)
    wave = np.append(hist, hist_rev_flip, 0)
    wave_str = " ".join([str(int(item[0])) for item in wave])

    if use_scene:
        scene = scene_detection.detect(input_img)
        audio_paths = SceneDetectionAudio.get_audio_for_scene("soundnet/categories_places2.txt", scene)
        print(audio_paths)

    data_audio = DataStructureAudio(
        root,
        scale,
        wave_str,
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


def convert_paintings_to_txt_bulk(input_dir, output_dir, with_saliency, use_scene):
    saliency_coarse = None
    if with_saliency:
        saliency_coarse = cv2.saliency.StaticSaliencySpectralResidual_create()
    scene_detection = None
    if use_scene:
        scene_detection = SceneDetectionVisual("resnet18")
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if ".jpg" in filename or ".png" in filename:
                input_file_path = os.path.join(root, filename)
                output_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
                convert_paintings_to_txt(input_file_path, output_file_path, saliency_coarse, with_saliency,
                                         scene_detection, use_scene)


def convert_paintings_to_txt(input_file_path, output_file_path, saliency_coarse, with_saliency, scene_detection,
                             use_scene):
    audio_data = scan_img(input_file_path, STEPS, saliency_coarse, with_saliency, scene_detection, use_scene)[0]
    audio_data.write_to_file(output_file_path)


def convert_txt_to_sound_bulk(exec_file, input_dir):
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            input_file_path = os.path.join(root, filename)
            output_filepath = os.path.join(root, f"{os.path.splitext(filename)[0]}.aiff")
            convert_txt_to_sound(exec_file, input_file_path, output_filepath)


def convert_txt_to_sound(exec_file, input_file_path, output_file_path):
    print(f"Begin: {os.path.splitext(input_file_path)[0]}")
    exec_path = os.path.abspath(exec_file)
    input_file_path = os.path.abspath(input_file_path)
    output_file_path = os.path.abspath(output_file_path)
    p = subprocess.Popen([
        "sclang",
        exec_path,
        input_file_path,
        output_file_path
    ])  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    p_status = p.wait()
    if p_status == 0:
        r = re.findall('(ERROR)+', str(output), flags=re.IGNORECASE)
        if r:
            print('An ERROR occurred!')
        else:
            print('File is executable!')
    print("End")


def convert_painting_to_presentation_bulk(input_dir, output_dir, with_saliency, use_scene, add_audio, web_convert,
                                          include_content, include_border):
    saliency_coarse = None
    if with_saliency:
        saliency_coarse = cv2.saliency.StaticSaliencySpectralResidual_create()
    scene_detection = None
    if use_scene:
        scene_detection = SceneDetectionVisual("resnet18")
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if ".jpg" in filename or ".png" in filename:
                file_path = os.path.join(root, filename)
                convert_painting_to_presentation(file_path, output_dir, saliency_coarse, with_saliency,
                                                 scene_detection, use_scene, add_audio,
                                                 web_convert, include_content, include_border)


def convert_painting_to_presentation(input_file_path, output_dir, saliency_coarse, with_saliency,
                                     scene_detection, use_scene, add_audio, web_convert,
                                     include_content, include_border):
    input_file_path = os.path.abspath(input_file_path)
    filename = os.path.basename(input_file_path)
    data = scan_img(input_file_path, STEPS, saliency_coarse, with_saliency, scene_detection, use_scene)

    visual_data = data[1]
    output_file_vid = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.avi")
    visual_data.generate_presentation_video(output_file_vid, include_content, include_border)
    if not add_audio and web_convert:
        convert_avi_to_webm(output_file_vid)

    if add_audio:
        audio_data = data[0]
        output_filepath_txt = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
        audio_data.write_to_file(output_filepath_txt)
        output_filepath_aiff = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.aiff")
        convert_txt_to_sound("sound_engine.scd", output_filepath_txt, output_filepath_aiff)
        output_file_vid_audio = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_audio.avi")
        add_audio_to_video(output_file_vid, output_filepath_aiff, output_file_vid_audio)
        if web_convert:
            convert_aiff_to_ogg(output_filepath_aiff)
            convert_avi_to_webm(output_file_vid_audio)


# ffmpeg -i yourvideo.avi -i sound.mp3 -c copy -map 0:v:0 -map 1:a:0 output.avi
# ffmpeg -i yourvideo.avi -i sound.aiff -c:v copy -c:a aac output.avi
def add_audio_to_video(input_vid, input_audio, output_vid):
    print("Begin adding audio")
    if not os.path.exists(input_vid):
        raise FileNotFoundError("Video file not found")
    if not os.path.exists(input_vid):
        raise FileNotFoundError("Audio file not found")

    p = subprocess.Popen([
        "ffmpeg",
        "-y",
        "-i",
        input_vid,
        "-i",
        input_audio,
        # "-shortest",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        output_vid
    ])  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    p_status = p.wait()
    if p_status == 0:
        r = re.findall('(ERROR)+', str(output), flags=re.IGNORECASE)
        if r:
            print('An ERROR occurred!')
        else:
            print('File is executable!')
    print("End")


# ffmpeg -i audio.wav -acodec libvorbis audio.ogg
def convert_aiff_to_ogg(input_file_path):
    print("Begin converting audio")
    if not os.path.exists(input_file_path):
        raise FileNotFoundError("Audio file not found")
    output_file_path = f"{os.path.splitext(input_file_path)[0]}.ogg"
    p = subprocess.Popen([
        "ffmpeg",
        "-y",
        "-i",
        input_file_path,
        "-acodec",
        "libvorbis",
        output_file_path
    ])  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    p_status = p.wait()
    if p_status == 0:
        r = re.findall('(ERROR)+', str(output), flags=re.IGNORECASE)
        if r:
            print('An ERROR occurred!')
        else:
            print('File is executable!')
    print("End")


def convert_avi_to_webm(input_file_path):
    print("Begin converting audio")
    if not os.path.exists(input_file_path):
        raise FileNotFoundError("Audio file not found")
    output_file_path = f"{os.path.splitext(input_file_path)[0]}.webm"
    p = subprocess.Popen([
        "ffmpeg",
        "-y",
        "-i",
        input_file_path,
        output_file_path
    ])  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    p_status = p.wait()
    if p_status == 0:
        r = re.findall('(ERROR)+', str(output), flags=re.IGNORECASE)
        if r:
            print('An ERROR occurred!')
        else:
            print('File is executable!')
    print("End")


class ImageConverter:
    pass
