import json
import os
import sys
import cv2

import torch

import logging

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
