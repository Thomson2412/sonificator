import os
import re
import subprocess
import ImageScanner

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

STEPS = 16


def convert_paintings_to_txt_bulk(input_dir, output_dir, saliency_coarse, with_saliency, scene_detection,
                                  use_object_nav, inner_scaling, things_as_chaos):
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if ".jpg" in filename or ".png" in filename:
                input_file_path = os.path.join(root, filename)
                output_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
                convert_paintings_to_txt(input_file_path, output_file_path, saliency_coarse, with_saliency,
                                         scene_detection, use_object_nav, inner_scaling, things_as_chaos)


def convert_paintings_to_txt(input_file_path, output_file_path, saliency_coarse, with_saliency, scene_detection,
                             use_object_nav, inner_scaling, things_as_chaos):
    data = ImageScanner.scan_img(
        input_file_path, STEPS, saliency_coarse, with_saliency, scene_detection,
        use_object_nav, inner_scaling, things_as_chaos)
    if data:
        data[0].write_to_file(output_file_path)


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


def convert_painting_to_presentation_bulk(input_dir, output_dir, sound_model, saliency_coarse, with_saliency,
                                          scene_detection, use_object_nav, add_audio, web_convert,
                                          include_content, include_border, inner_scaling, things_as_chaos):
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if ".jpg" in filename or ".png" in filename:
                file_path = os.path.join(root, filename)
                convert_painting_to_presentation(file_path, output_dir, sound_model, saliency_coarse, with_saliency,
                                                 scene_detection, use_object_nav, add_audio, web_convert,
                                                 include_content, include_border, inner_scaling, things_as_chaos)


def convert_painting_to_presentation(input_file_path, output_dir, sound_model, saliency_coarse, with_saliency,
                                     scene_detection, use_object_nav, add_audio, web_convert,
                                     include_content, include_border, inner_scaling, things_as_chaos):
    input_file_path = os.path.abspath(input_file_path)
    filename = os.path.basename(input_file_path)
    data = ImageScanner.scan_img(input_file_path, STEPS, saliency_coarse, with_saliency, scene_detection,
                                 use_object_nav, inner_scaling, things_as_chaos)

    if data:
        visual_data = data[1]
        output_file_vid = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.avi")
        visual_data.generate_presentation_video(output_file_vid, include_content, include_border)
        if not add_audio and web_convert:
            convert_avi_to_mp4(output_file_vid)

        if add_audio:
            audio_data = data[0]
            output_filepath_txt = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
            audio_data.write_to_file(output_filepath_txt)
            output_filepath_aiff = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.aiff")
            convert_txt_to_sound(sound_model, output_filepath_txt, output_filepath_aiff)
            output_file_vid_audio = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_audio.avi")
            add_audio_to_video(output_file_vid, output_filepath_aiff, output_file_vid_audio)
            if web_convert:
                convert_aiff_to_mp3(output_filepath_aiff)
                convert_avi_to_mp4(output_file_vid_audio)


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


# ffmpeg -i myinput.aif -f mp3 -acodec libmp3lame -ab 320000 -ar 44100 myoutput.mp3
def convert_aiff_to_mp3(input_file_path):
    print("Begin converting audio")
    if not os.path.exists(input_file_path):
        raise FileNotFoundError("Audio file not found")
    output_file_path = f"{os.path.splitext(input_file_path)[0]}.mp3"
    p = subprocess.Popen([
        "ffmpeg",
        "-y",
        "-i",
        input_file_path,
        "-acodec",
        "libmp3lame",
        "-ab",
        "320000",
        "-ar",
        "44100",
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


def convert_avi_to_mp4(input_file_path):
    print("Begin converting audio")
    if not os.path.exists(input_file_path):
        raise FileNotFoundError("Audio file not found")
    output_file_path = f"{os.path.splitext(input_file_path)[0]}.mp4"
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
