import os
import cv2
import numpy as np

input_folder = "data/impressionism/"


for root, dirs, files in os.walk(input_folder):
    for filename in files:
        file_path = os.path.join(root, filename)
        img = cv2.imread(file_path)
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(img)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        cv2.imwrite(f"{file_path.split('.')[0]}_saliency.{file_path.split('.')[1]}", saliencyMap)

        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(img)
        threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imwrite(f"{file_path.split('.')[0]}_saliency_fine.{file_path.split('.')[1]}", saliencyMap)
        cv2.imwrite(f"{file_path.split('.')[0]}_saliency_thresh.{file_path.split('.')[1]}", threshMap)

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