import logging
import os

import cv2
import numpy as np
import detectron2
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

from utils import save_img

setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances, register_coco_panoptic, register_coco_panoptic_separated


def get_bbox(mask):
    """
    Get bounding box of the mask
    :param mask:
    :return: [x, y, w, h]
    """
    loc = np.where(mask == 255)
    x = np.min(loc[1])
    y = np.min(loc[0])
    w = np.max(loc[1]) - x
    h = np.max(loc[0]) - y
    return [x, y, w, h]


def get_area(mask):
    """
    Get area of the mask
    :param mask:
    :return: area
    """
    return np.sum(mask == 255)


def get_segmented_img(img, masks):
    # colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
    # set of non repeating colors
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
    segmented_img = np.zeros(img.shape)
    for i in range(len(masks)):
        mask = masks[i]
        color = colors[i % len(colors)]
        segmented_img[mask == 255] = color
    return segmented_img


class ThresholdSegmentation():
    def __init__(self):
        pass

    def segment(self, img):
        """
        Segments red, green and blue objects from the image
        :param img: image to segment
        :return: segmented image, mask of each object, json containing instance details of each object
        """
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        instances = []

        # Extract red color mask
        # lower mask (0-10)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        # join my masks
        red_mask = mask0 + mask1
        instances += self.get_instance_mask(red_mask)

        # Extract green color mask
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([70, 255, 255])
        green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
        instances += self.get_instance_mask(green_mask)

        # Extract blue color mask
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
        instances += self.get_instance_mask(blue_mask)

        json = {}
        for i in range(len(instances)):
            mask = instances[i]
            instance_id = i + 1
            json[instance_id] = {}
            json[instance_id]["mask"] = mask.tolist()
            json[instance_id]["bbox"] = get_bbox(mask)
            json[instance_id]["area"] = get_area(mask)

        segmented_img = get_segmented_img(img, instances)
        return segmented_img, instances, json

    def get_instance_mask(self, red_mask):
        instances = []
        red_instances = cv2.connectedComponentsWithStats(red_mask, 4, cv2.CV_32S)

        (numLabels, labels, stats, centroids) = red_instances
        for i in range(1, numLabels):
            m = np.zeros(red_mask.shape, dtype="uint8")
            m[labels == i] = 255
            instances.append(m)
        return instances


class CocoPanopticSegmentation:
    def __init__(self, model_save_path=None):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.cfg.OUTPUT_DIR = os.path.join("output",
                                           self.__class__.__name__) if model_save_path is None else model_save_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
        self.predictor = DefaultPredictor(self.cfg)
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(self, coco_dataset_train_path, coco_dataset_val_path,
              train_json_path, val_json_path, coco_train_annotations_path, coco_train_segment_path):
        """
        Train the model
        """
        train_datset = "panoptic_custom_seg_train"
        val_dataset = "coco_seg_val"
        metadata = {
            "thing_classes": ['object', 'nature', 'sky', 'human', 'vehicle'],
            "thing_dataset_id_to_contiguous_id": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
            "stuff_dataset_id_to_contiguous_id": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
        }
        register_coco_panoptic_separated(train_datset, panoptic_json=train_json_path,
                                         panoptic_root=coco_train_annotations_path,
                                         image_root=coco_dataset_train_path,
                                         metadata={}, sem_seg_root=coco_train_segment_path,
                                         instances_json=train_json_path)
        # register_coco_instances(val_dataset, {}, val_json_path, coco_dataset_val_path)
        self.cfg.DATASETS.TRAIN = (train_datset,)
        # self.cfg.DATASETS.TEST = (val_dataset,)
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025
        self.cfg.SOLVER.MAX_ITER = 5000
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        # evaluator = COCOEvaluator(val_dataset, output_dir=self.cfg.OUTPUT_DIR)
        # trainer.test(self.cfg, trainer.model, evaluators=[evaluator])

    def segment(self, img):
        """
        Segments objects from the image
        :param img: image to segment
        :return: segmented image, mask of each object, json containing instance details of each object
        """
        panoptic_seg, segments_info = self.predictor(img)["panoptic_seg"]
        panoptic_seg = panoptic_seg.cpu().numpy()

        instances = []
        for i in range(len(segments_info)):
            mask = np.zeros(panoptic_seg.shape, dtype="uint8")
            mask[panoptic_seg == segments_info[i]["id"]] = 255
            instances.append(mask)

        json = {}
        for i in range(len(instances)):
            mask = instances[i]
            instance_id = i + 1
            json[instance_id] = {}
            # json[instance_id]["mask"] = mask.tolist()
            json[instance_id]["bbox"] = get_bbox(mask)
            json[instance_id]["area"] = get_area(mask)

        segmented_img = get_segmented_img(img, instances)
        return segmented_img, instances, json


if __name__ == '__main__':
    dlseg = CocoPanopticSegmentation()
    img = cv2.imread("coco_test.jpg")
    segmented_img, instances, json = dlseg.segment(img)
    # save_img(segmented_img, instances, json, "coco_test.jpg", "output")

    coco_path = "C:\\Users\\eshrestha\\Documents\\kitti\\semantics\\coco\\training\\annotations\\panoptic_kitti_training_2015.json"
    coco_dataset_path = "C:\\Users\\eshrestha\\Documents\\kitti\\semantics\\coco\\training\\panoptic_kitti_training_2015"
    coco_train_annotations_path = "C:\\Users\\eshrestha\\Documents\\kitti\\semantics\\coco\\training\\annotations\\panoptic_kitti_training_2015"
    coco_seg_path = "C:\\Users\\eshrestha\\Documents\\kitti\\semantics\\coco\\training\\segments"

    dlseg.train(train_json_path=coco_path, val_json_path=coco_path, coco_dataset_train_path=coco_dataset_path,
                coco_train_annotations_path=coco_train_annotations_path,
                coco_dataset_val_path=coco_dataset_path, coco_train_segment_path=coco_seg_path)
