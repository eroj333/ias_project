import json
import logging
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio.v2 as io
from tqdm import tqdm

CITYSCAPES_LABELS_MAP = {
    'unlabeled': 0,
    'ego': 1,
    'rectification': 2,
    'out roi': 3,
    'static': 4,
    'dynamic': 5,
    'ground': 6,
    'road': 7,
    'sidewalk': 8,
    'parking': 9,
    'rail': 10,
    'building': 11,
    'wall': 12,
    'fence': 13,
    'guard': 14,
    'bridge': 15,
    'tunnel': 16,
    'pole': 17,
    'polegroup': 18,
    'traffic light': 19,
    'traffic sign': 20,
    'vegetation': 21,
    'terrain': 22,
    'sky': 23,
    'person': 24,
    'rider': 25,
    'car': 26,
    'truck': 27,
    'bus': 28,
    'caravan': 29,
    'trailer': 30,
    'train': 31,
    'motorcycle': 32,
    'bicycle': 33,
    'license': 34,
}

CITYSCAPES_IDX_TO_TEXT = {v: k for k, v in CITYSCAPES_LABELS_MAP.items()}
CITYSCAPES_LABELS = [k for k, v in CITYSCAPES_LABELS_MAP.items()]
# CITYSCAPES_IDX_TO_KITTI_IDX = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8]
CITYSCAPES_IDX_TO_KITTI_IDX = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 7, 7,
                               7, 7, 7, 7, 7]


def get_file_list(src_dir):
    return [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]


class KittiToCocoPanopticConverter:
    def __init__(self, kitti_root, output_dir):
        self.kitti_root = kitti_root
        self.output_dir = output_dir
        # self.category_idx_map = {'__background__': 0, 'Car': 1, 'Van': 2, 'Truck': 3,
        #                          'Pedestrian': 4, 'Person_sitting': 5, 'Cyclist': 6,
        #                          'Tram': 7, 'Misc': 8, 'DontCare': 9}
        self.category_idx_map = {'void': 0, 'flat': 1, 'construction': 2, 'object': 3, 'nature': 4, 'sky': 5,
                                 'human': 6, 'vehicle': 7}
        self.cat_from_idx = {v: k for k, v in self.category_idx_map.items()}
        # self.category_color_map = {'__background__': [0, 0, 0], 'Car': [0, 0, 142], 'Van': [0, 0, 142],
        #                            'Truck': [0, 0, 142],
        #                            'Pedestrian': [220, 20, 60], 'Person_sitting': [220, 20, 60],
        #                            'Cyclist': [119, 11, 32],
        #                            'Tram': [0, 0, 142], 'Misc': [0, 0, 142], 'DontCare': [0, 0, 0]}
        self.category_color_map = {'void': [0, 0, 0], 'flat': [0, 0, 142], 'construction': [0, 0, 142],
                                   'object': [153, 153, 153], 'nature': [107, 142, 35], 'sky': [0, 0, 255],
                                   'human': [220, 20, 60], 'vehicle': [0, 60, 100]}
        self.catidx2color = {v: self.category_color_map[k] for k, v in self.category_idx_map.items()}
        self.categories_text = [k for k, v in self.category_idx_map.items()]

    def convert(self):
        kitti_train_dir = os.path.join(self.kitti_root, "training")
        kitti_val_dir = os.path.join(self.kitti_root, "testing")

        self.convert_kitti(kitti_train_dir, "training")
        # self.convert_kitti(kitti_val_dir, "testing")

    def convert_kitti(self, src_dir, dataset_name):

        annotation_name = "panoptic_kitti_{}_{}".format(dataset_name, "2015")

        kitti_image_dir = os.path.join(src_dir, "image_2")
        kitti_instance_dir = os.path.join(src_dir, "instance")
        kitti_label_dir = os.path.join(src_dir, "semantic")
        kitti_label_color_dir = os.path.join(src_dir, "semantic_rgb")

        kitti_image_files = get_file_list(kitti_image_dir)
        kitti_instance_files = get_file_list(kitti_instance_dir)
        kitti_label_files = get_file_list(kitti_label_dir)
        kitti_label_color_files = get_file_list(kitti_label_color_dir)

        kitti_image_files.sort()
        kitti_instance_files.sort()
        kitti_label_files.sort()
        kitti_label_color_files.sort()

        info = {
            "description": "KITTI Panoptic Dataset",
            "url": "http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015",
            "version": "1.0",
            "year": 2015,
            "contributor": "KITTI",
            "date_created": "2020/12/12"
        }

        licenses = {}

        annotations = []
        images = []
        categories = []

        # initialize progress bar
        logging.info("Converting KITTI to COCO Panoptic format")

        for k, v in self.category_idx_map.items():
            categories.append({"id": v, "name": k, "supercategory": k,
                               "isthing": 1 if k != "__background__" or k != "DontCare" else 0,
                               "color": self.catidx2color[v]})

        itr = tqdm(range(len(kitti_image_files)))
        itr.set_description("Converting KITTI to COCO Panoptic format")
        for image, lbl, instance_file in zip(kitti_image_files, kitti_label_files, kitti_instance_files):
            jpg_img = image.replace(".png", ".jpg")
            current_instance_id = 0
            itr.set_postfix_str(f"Processing {image}")
            itr.update(1)
            assert image == lbl == instance_file
            image_id = int(image.split(".")[0])
            image_file = os.path.join(kitti_image_dir, image)
            gt_img = cv2.imread(image_file)
            image_height, image_width, image_channels = gt_img.shape
            image_info = {
                "id": image_id,
                "width": image_width,
                "height": image_height,
                "file_name": jpg_img
            }
            images.append(image_info)

            gt_seg_img = np.zeros_like(gt_img)

            segments_info = []

            lbl_file = os.path.join(kitti_label_dir, lbl)
            # lbl_img = io.imread(lbl_file)

            instance_file_path = os.path.join(kitti_instance_dir, instance_file)
            instance_img = io.imread(instance_file_path)

            lbl_img = (instance_img >> 8) & 0xFF
            instance_img = instance_img & 0xFF

            # print("Instances:", np.unique(instance_img))

            # plt.imshow(instance_img)
            # plt.title("instance")
            # plt.show()

            # print(np.unique(lbl_img))
            # plt.imshow(lbl_img)
            # plt.title("Label")
            # plt.show()

            num_instances = np.unique(instance_img)
            num_instances.sort()
            if 0 in num_instances:
                num_instances = num_instances[1:]

            def create_segment(mask, instance_id, cityscapes_category_id, im_id):
                mask_x, mask_y = np.where(mask == 255)
                if len(mask_x) == 0:
                    return None
                x_min = int(np.min(mask_x))
                y_min = int(np.min(mask_y))

                x_max = int(np.max(mask_x))
                y_max = int(np.max(mask_y))

                x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

                if w == 0 or h == 0:
                    return None

                category_id = CITYSCAPES_IDX_TO_KITTI_IDX[cityscapes_category_id]
                # print(category_id, CITYSCAPES_IDX_TO_TEXT[cityscapes_category_id], self.cat_from_idx[category_id])

                segment_info = {"id": int(instance_id),
                                "category_id": int(category_id),
                                "area": w * h,
                                "iscrowd": 0,
                                "bbox": [x, y, w, h],
                                "image_id": int(im_id)}
                return segment_info

            new_instance_img = np.zeros(instance_img.shape)
            for instance_id in num_instances:
                # print(instance_id, type(instance_img))

                combined_instances = np.where(instance_img == instance_id, 255, 0)
                lbl_combined = np.where(instance_img == instance_id, lbl_img, 0)

                unique_classes = np.unique(lbl_combined)
                # print("Unique classes:", unique_classes)

                for c in unique_classes:
                    if c != 0:
                        mask = np.where(lbl_combined == c, 255, 0)
                        # print(mask.shape, np.sum(mask))
                        new_instance_img[mask == 255] = current_instance_id
                        segment_info = create_segment(mask, current_instance_id, c, image_id)

                        cat_color = self.category_color_map[self.cat_from_idx[CITYSCAPES_IDX_TO_KITTI_IDX[c]]]
                        gt_seg_img[mask == 255] = cat_color

                        if segment_info is not None:
                            segments_info.append(segment_info)
                        # print(segment_info)
                        current_instance_id += 1

            annotation = {"id": len(annotations) + 1, "image_id": image_id, "file_name": jpg_img,
                          "segments_info": segments_info}

            annotations.append(annotation)

            # write instance and label image
            instance_dest_path = os.path.join(self.output_dir, dataset_name, "annotations", annotation_name, jpg_img)
            os.makedirs(os.path.dirname(instance_dest_path), exist_ok=True)
            cv2.imwrite(instance_dest_path, new_instance_img)

            img_dest_path = os.path.join(self.output_dir, dataset_name, annotation_name, jpg_img)
            os.makedirs(os.path.dirname(img_dest_path), exist_ok=True)
            cv2.imwrite(img_dest_path, gt_img)

            seg_dest_path = os.path.join(self.output_dir, dataset_name, "segments", jpg_img)
            os.makedirs(os.path.dirname(seg_dest_path), exist_ok=True)
            cv2.imwrite(seg_dest_path, gt_seg_img)

        coco_panoptic_json = {"info": info, "licenses": licenses, "annotations": annotations, "images": images,
                              "categories": categories}

        json_dest_path = os.path.join(self.output_dir, dataset_name, "annotations", f"{annotation_name}.json")
        os.makedirs(os.path.dirname(json_dest_path), exist_ok=True)
        with open(json_dest_path, "w") as f:
            json.dump(coco_panoptic_json, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", type=str, required=False, help="Path to KITTI root directory",
                        default="C:\\Users\\eshrestha\\Documents\\kitti\\semantics")
    parser.add_argument("--output_dir", type=str, required=False, help="Path to output directory",
                        default="C:\\Users\\eshrestha\\Documents\\kitti\\semantics\\coco")

    args = parser.parse_args()

    converter = KittiToCocoPanopticConverter(args.kitti_root, args.output_dir)
    converter.convert()
