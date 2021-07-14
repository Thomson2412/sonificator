import os
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class ObjectDetectionVisual:

    def __init__(self):
        # Inference with a panoptic segmentation model
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

    def detect(self, file_path):
        img = cv2.imread(file_path)
        panoptic_seg, segments_info = self.predictor(img)["panoptic_seg"]
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        return out.get_image()[:, :, ::-1]

    def detect_bulk(self, input_dir):
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                if ".jpg" in filename or ".png" in filename:
                    input_file_path = os.path.join(root, filename)
                    result = self.detect(input_file_path)
                    cv2.imshow('image', result)
                    cv2.waitKey(0)
