import logging

import cv2
import numpy as np
import pandas as pd
import torch
import urllib.request

import matplotlib.pyplot as plt
from pyntcloud import PyntCloud

logger = logging.getLogger(__name__)

class MidasDepthEstimator():
    def __init__(self, model_type='large', device=None):
        """
        :param model_type: model type: l, s
        :param model_path: path to saved model
        :param device: device to run the model
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        # check if model_type is valid
        if model_type not in ['l', 's']:
            raise ValueError(f"Invalid model_type: {model_type}")

        # model type mapping
        model_type_mapping = {'l': 'DPT_Large', 's': 'MiDaS_small'}
        model_type = model_type_mapping[model_type]

        self.estimator = torch.hub.load("intel-isl/MiDaS", model_type)
        self.estimator.to(device)
        self.estimator.eval()
        self.logger.info(f"Loaded model: {model_type}")

    def estimate(self, img):
        """
        Estimate depth from image
        :param img: image to estimate depth
        :return: depth map
        """
        self.logger.info("Estimating depth")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img / 255.).float()
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        with torch.no_grad():
            depth = self.estimator(img)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=img.shape[2:],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = depth.cpu().numpy()
        self.logger.debug("Depth estimated")
        return depth


def visualize(img, depth):
    """
    Visualize depth map
    :param img: image
    :param depth: depth map
    """
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(depth)
    plt.show()

def visualize_3d(img, depth_map):
    """
    Visualize image in 3d
    """
    logger.info("Visualizing 3d")
    baseline = 0.1  # Adjust this value as per your setup

    # Calculate the focal length (you may need to adjust this as well)
    focal_length = 500  # Adjust this value as per your camera calibration

    # Create a grid of 3D points
    logger.debug("Creating 3d points")
    height, width = depth_map.shape
    y, x = np.mgrid[0:height, 0:width]
    x3d = (x - width / 2) * depth_map / focal_length
    y3d = (y - height / 2) * depth_map / focal_length
    z3d = depth_map

    # Apply the baseline to the x-coordinate for stereo disparity
    x3d += baseline

    # Visualize the 3D representation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Display the 3D points as a scatter plot
    logger.debug("Displaying 3d points")
    ax.scatter(x3d, -y3d, z3d, c=img)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()





if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # download image from url
    logger.debug("Downloading image")
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)

    # load image
    logger.debug("Loading image")
    img = cv2.imread(filename)

    # create depth estimator
    logger.debug("Creating depth estimator")
    depth_estimator = MidasDepthEstimator(model_type='l', device=torch.device('cpu'))

    # estimate depth
    depth = depth_estimator.estimate(img)

    # visualize depth
    # visualize(img, depth)

    visualize_3d(img, depth)