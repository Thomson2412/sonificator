import os
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
import Utils


def main():
    input_folder = "../data/new_dataset_scene_correct_resize_segments/"
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if "_object_segments" in filename or "_object_colors" in filename:
                continue
            print(f"Working on: {filename}")
            file_path = os.path.abspath(os.path.join(root, filename))

            img = cv2.imread(file_path)
            if img is None:
                continue

            cfg = get_cfg()
            if not torch.cuda.is_available():
                cfg.MODEL.DEVICE = "cpu"
            cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
            predictor = DefaultPredictor(cfg)
            panoptic_seg, segments_info = predictor(img)["panoptic_seg"]
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
            out_img = out.get_image()[:, :, ::-1]

            filename_segments = os.path.join(
                root,
                f"{os.path.splitext(filename)[0]}_object_segments{os.path.splitext(filename)[1]}")
            cv2.imwrite(filename_segments, out_img)

            color_segments_img = np.array(img, copy=True)
            for mask_id in np.unique(panoptic_seg.to("cpu")):
                mask = panoptic_seg.to("cpu") == mask_id
                sub_img_reshape = img[mask]
                dominant_color = Utils.get_dominant_color(sub_img_reshape, 2)
                dominant_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV).flatten()
                color_segments_img[mask] = cv2.cvtColor(np.uint8([[dominant_hsv]]), cv2.COLOR_HSV2BGR)

            filename_colors = os.path.join(
                root,
                f"{os.path.splitext(filename)[0]}_object_colors{os.path.splitext(filename)[1]}")
            cv2.imwrite(filename_colors, color_segments_img)

            print(f"Done: {filename}")


if __name__ == '__main__':
    main()
