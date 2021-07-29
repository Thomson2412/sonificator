import os
import SceneDetectionAudio
from SceneDetectionVisual import SceneDetectionVisual
from shutil import copyfile


def main():
    input_folder_img = "/mnt/datadrive/projects/thesis/Datasets/Paintings/painter_by_numbers_scene_correct/"
    audio_dataset_base_dir = "/mnt/datadrive/projects/thesis/Datasets/Audio/"
    output_folder = "../data/scene_audio/"
    scene_detection = SceneDetectionVisual("../places", "resnet18")

    for root, dirs, files in os.walk(input_folder_img):
        for filename in files:
            print(f"Working on: {filename}")
            file_path = os.path.join(root, filename)
            scene = scene_detection.detect(file_path).replace("/", "_")
            output_path = os.path.join(output_folder, scene)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
                scene_audio_paths = SceneDetectionAudio.get_audio_for_scene("../soundnet/audio_scene_detection.json",
                                                                            audio_dataset_base_dir,
                                                                            scene)
                for audio_path in scene_audio_paths:
                    filename_audio = os.path.basename(audio_path)
                    output_file_path = os.path.join(output_path, filename_audio)
                    copyfile(audio_path, output_file_path)

            print(f"Done: {filename}")


if __name__ == '__main__':
    main()
