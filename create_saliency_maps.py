import os
import cv2
import numpy as np


def main():
    input_folder = "data/impressionism_saliency/"
    file_blocked_list = ["9989.jpg", "81823.jpg"]

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    saliency_fine = cv2.saliency.StaticSaliencyFineGrained_create()

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if "_" in filename or filename in file_blocked_list:
                continue
            print(f"Working on: {filename}")
            file_path = os.path.join(root, filename)
            img = None

            filename_saliency = f"{file_path.split('.')[0]}_saliency.{file_path.split('.')[1]}"
            if not os.path.exists(filename_saliency):
                if img is None:
                    img = cv2.imread(file_path)
                (success, saliency_map) = saliency.computeSaliency(img)
                saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                heat_img = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
                add_img = cv2.addWeighted(img, 0.3, heat_img, 0.7, 0)
                cv2.imwrite(filename_saliency, add_img)

            filename_saliency_fine = f"{file_path.split('.')[0]}_saliency_fine.{file_path.split('.')[1]}"
            filename_thresh = f"{file_path.split('.')[0]}_saliency_thresh.{file_path.split('.')[1]}"
            if not os.path.exists(filename_saliency_fine) or not os.path.exists(filename_thresh):
                if img is None:
                    img = cv2.imread(file_path)
                (success, saliency_map) = saliency_fine.computeSaliency(img)
                saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                heat_img = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
                add_img = cv2.addWeighted(img, 0.3, heat_img, 0.7, 0)
                cv2.imwrite(filename_saliency_fine, add_img)
                thresh_map = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                cv2.imwrite(filename_thresh, thresh_map)

            print(f"Done: {filename}")

            # saliency = cv2.saliency.ObjectnessBING_create()
            # saliency.setTrainingPath("model")
            #
            # (success, saliencyMap) = saliency.computeSaliency(img)
            # numDetections = saliencyMap.shape[0]
            # # loop over the detections
            # for i in range(0, min(numDetections, 6)):
            #     # extract the bounding box coordinates
            #     (startX, startY, endX, endY) = saliencyMap[i].flatten()
            #
            #     # randomly generate a color for the object and draw it on the image
            #     output = img.copy()
            #     color = np.random.randint(0, 255, size=(3,))
            #     color = [int(c) for c in color]
            #     cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
            #     # show the output image
            #     cv2.imshow("Image", output)
            #     cv2.waitKey(0)


if __name__ == '__main__':
    main()
