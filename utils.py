import json
import os
import random
import sys
import cv2
import numpy as np

import torch

import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.utils.visualizer import Visualizer

from pycocotools import mask as coco_mask
from skimage import measure

logger = logging.getLogger(__name__)


def load_model(model_path):
    """
    Load model from path
    """
    logger.debug(f"Loading model from {model_path}")
    if not os.path.isfile(model_path):
        print("Model does not exist")
        sys.exit(0)
    model = torch.load(model_path)
    model.eval()
    logger.debug("Model loaded")
    return model


def load_img(img_path):
    """
    Load image from path
    """
    logger.debug(f"Loading image from {img_path}")
    if not os.path.isfile(img_path):
        print("Image does not exist")
        sys.exit(0)
    img = cv2.imread(img_path)
    logger.debug("Image loaded")
    return img


def save_img(img, mask, json_dict, depth_map, original_img_path, path):
    """
    Save segmentation details
    :param img: segmented image
    :param mask: list of masks
    :param json_dict: dict containing instance details of each mask
    :param depth_map: depth map
    :param original_img_path: original image path
    :param path: save destination
    """
    logger.info("Saving segmentation details")


    # save segmented image as dest_path + orginal_image_name/segmented.png
    logger.debug("Saving segmented image")
    img_name = os.path.basename(original_img_path).split(".")[0]
    img_path = os.path.join(path, img_name, "segmented.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    cv2.imwrite(img_path, img)

    # save mask images as dest_path + orginal_image_name/mask + mask_id + .png
    logger.debug("Saving mask images")
    for i in range(len(mask)):
        mask_path = os.path.join(path, img_name, "mask", str(i) + ".png")
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, mask[i])

    depth_path = os.path.join(path, img_name, "depth.png")
    os.makedirs(os.path.dirname(depth_path), exist_ok=True)
    cv2.imwrite(depth_path, depth_map)
    # save json as dest_path + orginal_image_name/segments.json
    # logger.debug("Saving json")
    # json_path = os.path.join(path, img_name, "segments.json")
    # os.makedirs(os.path.dirname(json_path), exist_ok=True)
    # with open(json_path, 'w') as f:
    #     json.dump(str(json_dict), f)
    # logger.info("Saved segmentation details")


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.uint32)):
            return obj.item()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        else:
            return super().default(obj)


def instance_segment_contours(mask_bin):
    fortran_ground_truth_binary_mask = np.asfortranarray(mask_bin)
    encoded_ground_truth = coco_mask.encode(fortran_ground_truth_binary_mask)
    area = coco_mask.area(encoded_ground_truth)
    bbox = coco_mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(mask_bin, 0.5)

    return area.tolist(), bbox.tolist(), [np.flip(contour, axis=1).ravel().tolist() for contour in contours]


def visualize_dataset_samples(dataset_name):
    """
    Visualize samples from dataset
    :param dataset_name: name of dataset
    """
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("", vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def register_dataset(dataset_name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root,
                     instances_json):
    register_coco_panoptic_separated(name=dataset_name,
                                     metadata=metadata,
                                     image_root=image_root,
                                     panoptic_root=panoptic_root,
                                     panoptic_json=panoptic_json,
                                     sem_seg_root=sem_seg_root,
                                     instances_json=instances_json)