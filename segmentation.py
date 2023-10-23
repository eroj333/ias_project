import cv2
import numpy as np


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
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
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


class Mask2FormerSegmentation():
    def __init__(self):
        pass

    def segment(self, img):
        """
        :param img: image to segment
        :return: segmented image, mask of each object, json containing instance details of each object
        """
