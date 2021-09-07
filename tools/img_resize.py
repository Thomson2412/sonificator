import os
import cv2


def main():
    input_folder_img = "/mnt/datadrive/projects/thesis/Datasets/Paintings/new_dataset_scene_correct/"
    output_folder = "/mnt/datadrive/projects/thesis/Datasets/Paintings/new_dataset_scene_correct_resize/"

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
            cv2.imwrite(os.path.join(output_folder, filename), img)

            print(f"Done: {filename}")


if __name__ == '__main__':
    main()
