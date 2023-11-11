from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.data import DatasetCatalog
from detectron2.utils.logger import setup_logger

import logging
import os
import cv2
import numpy as np

setup_logger()


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


def get_segmented_img(img, masks, seg_info=None):
    # colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
    # set of non repeating colors
    # colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
    colors = [[220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], [0, 60, 100], [0, 80, 100],
              [0, 0, 70], [0, 0, 192], [250, 170, 30], [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30],
              [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157], [110, 76, 0],
              [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240], [0, 125, 92], [209, 0, 151],
              [188, 208, 182], [0, 220, 176], [255, 99, 164], [92, 0, 73], [133, 129, 255], [78, 180, 255], [0, 228, 0],
              [174, 255, 243], [45, 89, 255], [134, 134, 103], [145, 148, 174], [255, 208, 186], [197, 226, 255],
              [171, 134, 1], [109, 63, 54], [207, 138, 255], [151, 0, 95], [9, 80, 61], [84, 105, 51], [74, 65, 105],
              [166, 196, 102], [208, 195, 210], [255, 109, 65], [0, 143, 149], [179, 0, 194], [209, 99, 106],
              [5, 121, 0], [227, 255, 205], [147, 186, 208], [153, 69, 1], [3, 95, 161], [163, 255, 0], [119, 0, 170],
              [0, 182, 199], [0, 165, 120], [183, 130, 88], [95, 32, 0], [130, 114, 135], [110, 129, 133],
              [166, 74, 118], [219, 142, 185], [79, 210, 114], [178, 90, 62], [65, 70, 15], [127, 167, 115],
              [59, 105, 106], [142, 108, 45], [196, 172, 0], [95, 54, 80], [128, 76, 255], [201, 57, 1], [246, 0, 122],
              [191, 162, 208]]
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

    def train(self, coco_dataset_train_images_path, coco_dataset_val_path,
              train_json_path, instances_json_path, coco_train_annotations_path, coco_train_segment_path):
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
        register_coco_panoptic_separated(name=train_datset,
                                         metadata={},
                                         image_root=coco_dataset_train_images_path,
                                         panoptic_root=coco_train_annotations_path,
                                         panoptic_json=train_json_path,
                                         sem_seg_root=coco_train_segment_path,
                                         instances_json=instances_json_path)
        # register_coco_instances(val_dataset, {}, val_json_path, coco_dataset_val_path)
        DatasetCatalog.get(train_datset + '_separated')

        self.cfg.DATASETS.TRAIN = (train_datset + '_separated',)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025
        self.cfg.SOLVER.MAX_ITER = 5000
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8

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
        classes = []
        for i in range(len(segments_info)):
            mask = np.zeros(panoptic_seg.shape, dtype="uint8")
            mask[panoptic_seg == segments_info[i]["id"]] = 255
            if segments_info[i]["isthing"]:
                instances.append(mask)
                classes.append(segments_info[i]["category_id"])

        json = {}
        for i in range(len(instances)):
            mask = instances[i]
            instance_id = i + 1
            json[instance_id] = {}
            # json[instance_id]["mask"] = mask.tolist()
            json[instance_id]["bbox"] = get_bbox(mask)
            json[instance_id]["area"] = get_area(mask)
            json[instance_id]["class"] = classes[i]

        segmented_img = get_segmented_img(img, instances, segments_info)
        return segmented_img, instances, json


if __name__ == '__main__':
    dlseg = CocoPanopticSegmentation()
    img = cv2.imread("coco_test.jpg")
    segmented_img, instances, json = dlseg.segment(img)
    # save_img(segmented_img, instances, json, "coco_test.jpg", "output")

    coco_panoptic_json_path = "C:\\Users\\eshrestha\\Documents\\kitti\\semantics\\coco\\training\\annotations\\panoptic_kitti_training_2015.json"
    coco_instances_json_path = "C:\\Users\\eshrestha\\Documents\\kitti\\semantics\\coco\\training\\panoptic_instances.json"
    coco_dataset_path = "C:\\Users\\eshrestha\\Documents\\kitti\\semantics\\coco\\training\\panoptic_kitti_training_2015"
    coco_train_annotations_path = "C:\\Users\\eshrestha\\Documents\\kitti\\semantics\\coco\\training\\annotations\\panoptic_kitti_training_2015"
    coco_seg_path = "C:\\Users\\eshrestha\\Documents\\kitti\\semantics\\coco\\training\\segments"

    coco_panoptic_json_path = "/mnt/c/Users/eshrestha/Documents/kitti/semantics/coco/training/annotations/panoptic_kitti_training_2015.json"
    coco_instances_json_path = "/mnt/c/Users/eshrestha/Documents/kitti/semantics/coco/training/panoptic_instances.json"
    coco_dataset_path = "/mnt/c/Users/eshrestha/Documents/kitti/semantics/coco/training/panoptic_kitti_training_2015"
    coco_train_annotations_path = "/mnt/c/Users/eshrestha/Documents/kitti/semantics/coco/training/annotations/panoptic_kitti_training_2015"
    coco_seg_path = "/mnt/c/Users/eshrestha/Documents/kitti/semantics/coco/training/segments"

    dlseg.train(train_json_path=coco_panoptic_json_path, instances_json_path=coco_instances_json_path,
                coco_dataset_train_images_path=coco_dataset_path,
                coco_train_annotations_path=coco_train_annotations_path,
                coco_dataset_val_path=coco_dataset_path, coco_train_segment_path=coco_seg_path)
