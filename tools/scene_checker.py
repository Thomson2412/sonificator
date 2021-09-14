import os
import cv2
from SceneDetectionVisual import SceneDetectionVisual
import shutil


def main():
    # input_folder_img = "/mnt/datadrive/projects/thesis/Datasets/Paintings/new_dataset/"
    # output_folder = "/mnt/datadrive/projects/thesis/Datasets/Paintings/new_dataset_scene_correct/"
    input_folder_img = "../data/test/temp_scene"
    output_folder = "../data/test/temp_scene"
    scene_detection = SceneDetectionVisual("../places", "resnet18")

    for root, dirs, files in os.walk(input_folder_img):
        for filename in files:
            print(f"Working on: {filename}")
            file_path = os.path.join(root, filename)

            img = cv2.imread(file_path)
            scale_h = (1080 / 2) / img.shape[0]
            width = int(img.shape[1] * scale_h)
            height = int(img.shape[0] * scale_h)
            dim = (width, height)
            img = cv2.resize(img, dim)

            scene = scene_detection.detect(file_path)

            print(scene)

            cv2.imshow("Scene", img)
            cv2.waitKey(20)
            user_input = input(f"{scene} correct?")

            if user_input == "y":
                shutil.copy2(file_path, output_folder)

            print(f"Done: {filename}")


if __name__ == '__main__':
    main()

