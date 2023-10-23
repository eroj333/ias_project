"""
Run panoptic segmentation on a single image
input: image path, model path, output path
output: panoptic segmentation image, mask images, json file
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import json
import logging
from segmentation import ThresholdSegmentation

# format log: <date> <time> <level> <module> <message>
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse input arguments
    """
    args = argparse.ArgumentParser(description='Panoptic Segmentation')
    args.add_argument('--image', help="path to image", default="test_img.png", type=str)
    # args.add_argument('--model', help="path to model", default="data/model.pt", type=str)
    args.add_argument('--model_path', help="path to model", default="data/model.pt", type=str)
    args.add_argument('--method', help="path to model", default="cv2", options=['cv2', 'm2f'], type=str)
    args.add_argument('--output', help="path to output", default="output", type=str)
    return args.parse_args()


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


def save_img(img, mask, json_dict, original_img_path, path):
    """
    Save segmentation details
    :param img: segmented image
    :param mask: list of masks
    :param json_dict: dict containing instance details of each mask
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

    # save json as dest_path + orginal_image_name/segments.json
    logger.debug("Saving json")
    json_path = os.path.join(path, img_name, "segments.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(str(json_dict), f)
    logger.info("Saved segmentation details")


def resolve_model(args):
    """
    Resolve model from input
    :param model: path to model or name of model
    :return: model
    """
    if args.method == "cv2":
        return ThresholdSegmentation()
    else:
        return None


def visualize(img, masks):
    """
    Visualize parts of image based on mask
    :param masks:
    :param img:
    """
    segments = img.copy()
    for mask in masks:
        img[mask != 255] = [0, 0, 0]
    cv2.imshow("", segments)


def main(args):
    img = load_img(args.image)
    segmentation_model = resolve_model(args)

    if segmentation_model is None:
        logger.error("Invalid model")
        sys.exit(0)

    segmented_img, masks, json = segmentation_model.segment(img)
    save_img(segmented_img, masks, json, args.image, args.output)
    visualize(img, masks)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)