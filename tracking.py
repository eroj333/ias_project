import cv2
import numpy as np


def binarize_masks(instance_masks):
    """
    Convert instance masks to binary masks. Removes masks with no pixels.
    """
    binary_masks = []
    for mask in instance_masks:
        binary_mask = np.array(mask > 0, dtype=np.uint8)
        if np.sum(binary_mask) != 0:
            binary_masks.append(binary_mask)
    return binary_masks


def compute_iou(mask, tracked_instance):
    """
    Compute IoU between two masks.
    """
    intersection = np.logical_and(mask, tracked_instance)
    union = np.logical_or(mask, tracked_instance)
    iou = np.sum(intersection) / np.sum(union)
    return iou


class HungarianTracker:
    def __init__(self):
        self.tracked_instances = []
        self.tracked_ids = []
        self.max_id = 0

    def track(self, instance_masks, img):
        """
        Update tracking information with new instance masks. Remove instances that are no longer present.
        """
        im_cp = img.copy()
        binary_masks = binarize_masks(instance_masks)
        updated_tracked_instances = [False] * len(self.tracked_instances)
        if len(self.tracked_instances) == 0:
            self.tracked_instances = binary_masks
            self.tracked_ids = list(range(len(binary_masks)))
            self.max_id = len(binary_masks)
        else:
            # Compute IoU with each tracked instance
            ious = []
            for tracked_instance in self.tracked_instances:
                ious.append([compute_iou(mask, tracked_instance) for mask in binary_masks])

            # Assign the mask to the tracked instance with highest IoU
            new_masks_id = set(range(len(binary_masks)))
            for i, iou in enumerate(ious):
                if iou is not None and sum(iou)>0 and max(iou) > 0.5:
                    track_id = iou.index(max(iou))
                    updated_tracked_instances[i] = True
                    self.tracked_instances[i] = binary_masks[track_id]
                    new_masks_id.remove(track_id)

            # Add new instances
            for i in new_masks_id:
                track_id = self.max_id
                self.tracked_instances.append(binary_masks[i])
                self.tracked_ids.append(track_id)
                self.max_id += 1
                updated_tracked_instances.append(True)

            # Remove instances that are no longer present
            self.tracked_instances = [self.tracked_instances[i] for i in range(len(self.tracked_instances)) if
                                      updated_tracked_instances[i]]
            self.tracked_ids = [self.tracked_ids[i] for i in range(len(self.tracked_ids)) if
                                updated_tracked_instances[i]]

        return self.tag_instances(im_cp)

    def tag_instances(self, img):
        """
        Tag instance masks with their corresponding IDs, and draw bounding boxes around them.
        """
        for i, mask in enumerate(self.tracked_instances):
            indices = np.where(mask == 1)
            img[indices] = [255, 255, 0]
            x1, y1 = np.min(indices[1]), np.min(indices[0])
            x2, y2 = np.max(indices[1]), np.max(indices[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, str(self.tracked_ids[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return img
