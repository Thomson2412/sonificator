import os
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# TODO: Remove when not needed anymore
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def detect_instances(img):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)

    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]


def detect_panoptic(img):
    cfg = get_cfg()
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    pred = predictor(img)
    panoptic_seg, segments_info = pred["panoptic_seg"]
    segment_img = panoptic_seg.to("cpu").numpy()

    return segment_img, segments_info


def detect_bulk(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for filename in sorted(files):
            if ".jpg" in filename or ".png" in filename:
                print(f"Begin: {filename}")
                input_file_path = os.path.join(root, filename)
                img = cv2.imread(input_file_path)
                scan_img_segment(img)

            break
        break

                # result_inst = detect_instances(img)
                # output_filepath_inst = os.path.join(
                #     "data/test/detectron/",
                #     f"{os.path.splitext(filename)[0]}_instances{os.path.splitext(filename)[1]}"
                # )
                # cv2.imwrite(output_filepath_inst, result_inst)
                #
                # result_seg = detect_panoptic(img)
                # output_filepath_seg = os.path.join(
                #     "data/test/detectron/",
                #     f"{os.path.splitext(filename)[0]}_segmentation{os.path.splitext(filename)[1]}"
                # )
                # cv2.imwrite(output_filepath_seg, result_seg)
                # print(f"End: {filename}")


def scan_img_segment(img):
    segmentation_img, segmentation_info = detect_panoptic(img)

    for step in range(max([item["id"] for item in segmentation_info])):
        sub_img = img[segmentation_img == step]
        print(sub_img)

    # return data_audio, data_visual


class ObjectDetectionVisual:
    pass
