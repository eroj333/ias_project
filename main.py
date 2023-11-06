"""
Run panoptic segmentation on a single image
input: image path, model path, output path
output: panoptic segmentation image, mask images, json file
"""

import argparse
import sys
import cv2
import logging

import numpy as np

from segmentation import ThresholdSegmentation, CocoPanopticSegmentation
from utils import load_img, save_img
import matplotlib.pyplot as plt
from depth_estimation import MidasDepthEstimator, visualize_3d
from tracking import HungarianTracker

# format log: <date> <time> <level> <module> <message>
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(module)s %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse input arguments
    """
    args = argparse.ArgumentParser(description='Panoptic Segmentation')
    args.add_argument('--image', help="path to image", default="coco_test.jpg", type=str)
    args.add_argument('--video', help="path to video", default="demo/demo_1.mp4", type=str)
    # args.add_argument('--model', help="path to model", default="data/model.pt", type=str)
    args.add_argument('--model_path', help="path to model", default="data/model.pt", type=str)
    args.add_argument('--method', help="path to model", default="cocopan", choices=['cv2', 'cocopan'], type=str)
    args.add_argument('--output', help="path to output", default="output", type=str)
    return args.parse_args()


def resolve_model(args):
    """
    Resolve model from input
    :param model: path to model or name of model
    :return: model
    """
    if args.method == "cv2":
        return ThresholdSegmentation()
    elif args.method == "cocopan":
        return CocoPanopticSegmentation()
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


def bird_view(masks, depth):
    # Create bird's eye view given instance masks and depth map
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
    bboxes = []
    canvas = np.zeros((depth.shape[1], int(np.max(depth)) + 1, 3))
    logger.debug(f"Created canvas of shape {canvas.shape}")
    for i, mask in enumerate(masks):
        indices = np.where(mask == 255)
        indices = np.array(indices).T
        print(indices.shape, depth.shape)
        d = np.array(depth[indices[:, 0], indices[:, 1]], dtype=np.uint8)
        d = np.expand_dims(d, axis=1)
        coords = np.hstack((indices, d))

        # project coords to bird's eye view
        coords = coords[:, 1:]
        bbox = [
            np.min(coords[:, 0]),
            np.min(coords[:, 1]),
            np.max(coords[:, 0]),
            np.max(coords[:, 1])
        ]
        bboxes.append(bbox)
        canvas[coords] = colors[i % len(colors)]

    plt.imshow(canvas)
    plt.show()
    return bboxes


def process_video(video_file, segmentation_model):
    cap = cv2.VideoCapture(video_file)
    out_file = video_file.split(".")[0] + "_out.mp4"
    seg_out_file = video_file.split(".")[0] + "_seg_out.mp4"

    video_writer = None
    seg_video_writer = None
    tracker = HungarianTracker()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if video_writer is None:
            video_writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))
            seg_video_writer = cv2.VideoWriter(seg_out_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))
        segmented_img, masks, json = segmentation_model.segment(frame)

        # track instances
        tracked_img = tracker.track(masks, frame)

        # write to video
        video_writer.write(tracked_img)
        seg_video_writer.write(segmented_img.astype(np.uint8))

    cap.release()
    if video_writer is not None:
        video_writer.release()
        seg_video_writer.release()
    cv2.destroyAllWindows()


def main(args):
    img = load_img(args.image)
    video_file = args.video

    segmentation_model = resolve_model(args)

    if segmentation_model is None:
        logger.error("Invalid model")
        sys.exit(0)

    if video_file and video_file != "":
        process_video(video_file, segmentation_model)
    elif img is not None:
        segmented_img, masks, json = segmentation_model.segment(img)

        # visualize(img, masks)
        depth_estimator = MidasDepthEstimator()
        depth = depth_estimator.estimate(img)

        logger.debug(f"Depth map shape: {depth.shape}, segmented image shape: {segmented_img.shape}")
        save_img(segmented_img, masks, json, depth, args.image, args.output)
        # bboxes = bird_view(masks, depth)
        # visualize_3d(img, depth)
        # tracker = HungarianTracker()
        #
        # for i in range(10):
        #     tracked_img = tracker.track(masks, img)
        #     # convert to RGB
        #     tracked_img = cv2.cvtColor(tracked_img, cv2.COLOR_BGR2RGB)
        #     #visualize with plt
        #     plt.imshow(tracked_img)
        #     plt.show()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
