import json
import os

import numpy as np
from nuimages import NuImages
from nuimages.utils import utils as nuim_utils
import cv2
import utils
from segmentation import CocoPanopticSegmentation
from utils import register_dataset


class NuscenesToCocoPanopticConverter:
    def __init__(self, nuscenes_root, dataset, output_dir, lim=-1):
        self.nuscenes_root = nuscenes_root
        self.dataset = dataset
        self.output_dir = output_dir if lim < 0 else f"{output_dir}_{lim}"
        self.nuim = NuImages(version=self.dataset, dataroot=self.nuscenes_root, verbose=True)
        self.category_color_map = {'animal': [0, 0, 0], 'flat': [0, 0, 142], 'human': [0, 0, 142],
                                   'movable_object': [153, 153, 153], 'static_object': [107, 142, 35],
                                   'vehicle': [0, 0, 255],
                                   }
        self.all_categories = self.get_all_categories()
        self.categories_to_id = {category['name']: category['id'] for category in self.all_categories}
        self.things_categories = set([cat['name'] for cat in self.all_categories]).difference(set(['flat']))
        self.lim = lim

    def convert(self):
        # loop over keyframes and create panoptic annotations
        images = []
        panoptic_annotations = []
        instance_annotations = []

        # get all the scenes
        l = 0
        instance_id = 0
        for sample_data in self.nuim.sample_data:
            if 0 < self.lim < l:
                break
            sample_record = self.nuim.get('sample_data', sample_data['token'])
            im_path = os.path.join(self.nuscenes_root, sample_record['filename'])
            # extract filename from im_path
            gt_filename = os.path.basename(im_path)
            gt_png_filename = gt_filename.replace(".jpg", ".png")

            if os.path.exists(im_path) and sample_record['is_key_frame']:
                # get all the annotations
                object_tokens, surface_tokens = self.nuim.list_anns(sample_data['sample_token'])
                if len(object_tokens) == 0:
                    continue

                semantic_mask, instance_mask = self.nuim.get_segmentation(sample_data['token'])
                instance_mask = np.zeros_like(instance_mask)
                semantic_mask = np.zeros_like(semantic_mask)

                image_info = {
                    'id': sample_data['token'],
                    'width': sample_record['width'],
                    'height': sample_record['height'],
                    'file_name': gt_filename,
                }
                images.append(image_info)

                l += 1

                panoptic_segments = []
                for object_token in object_tokens:
                    panoptic_instance_id = len(panoptic_segments) + 1
                    ann = self.nuim.get('object_ann', object_token)
                    obj_cat = self.categories_to_id[
                        self.get_root_category(self.nuim.get('category', ann['category_token'])['name'])]
                    if ann['mask'] is None:
                        continue
                    instance_id += 1
                    mask_bin = nuim_utils.mask_decode(ann['mask'])
                    semantic_mask[mask_bin > 0] = obj_cat
                    instance_mask[mask_bin > 0] = panoptic_instance_id
                    segment_info = self.create_panoptic_segment(mask_bin,
                                                                instance_id=panoptic_instance_id,
                                                                im_id=sample_data['token'],
                                                                category_id=obj_cat)

                    _area, _bbox, _contours = utils.instance_segment_contours(mask_bin)

                    segment_info['area'] = _area
                    segment_info['bbox'] = _bbox

                    instance_ann = {"id": instance_id, #len(instance_annotations) + 1,
                                    "image_id": sample_data['token'],
                                    "category_id": obj_cat,
                                    "area": _area, #ann['bbox'][1] * ann['bbox'][2],
                                    "iscrowd": 0,
                                    "bbox": _bbox, #ann['bbox'],
                                    "segmentation": _contours, #utils.instance_segment_contours(mask_bin),
                                    # "segmentation": ann['mask'],
                                    "file_name": gt_png_filename
                                    }
                    instance_annotations.append(instance_ann)
                    if segment_info is not None:
                        panoptic_segments.append(segment_info)

                # for surface_token in surface_tokens:
                #     ann = self.nuim.get('surface_ann', surface_token)
                #     obj_cat = self.categories_to_id[
                #         self.get_root_category(self.nuim.get('category', ann['category_token'])['name'])]
                #     if ann['mask'] is None:
                #         continue
                #     instance_id += 1
                #     mask_bin = nuim_utils.mask_decode(ann['mask'])
                #     semantic_mask[mask_bin > 0] = obj_cat

                panoptic_ann = {
                    "id": len(panoptic_annotations) + 1,
                    "image_id": sample_data['token'],
                    "file_name": gt_png_filename,
                    "segments_info": panoptic_segments
                }
                panoptic_annotations.append(panoptic_ann)

                # save images
                im_path = os.path.join(self.nuscenes_root, sample_record['filename'])
                gt = cv2.imread(im_path)

                self.save_im(os.path.join(self.output_dir, self.dataset, "images", gt_filename), gt)
                self.save_im(os.path.join(self.output_dir, self.dataset, "semantic", gt_png_filename),
                             semantic_mask)
                self.save_im(os.path.join(self.output_dir, self.dataset, "panoptic_" + self.dataset, gt_png_filename),
                             instance_mask)

        # save annotations
        info = {"description": "NuScenes Panoptic Segmentation",
                "url": "https://www.nuscenes.org/",
                "version": self.dataset,
                "year": 2020,
                "contributor": "nuTonomy",
                "date_created": "2020/09/09"}

        coco_panoptic_json = {"info": info, "licenses": {}, "annotations": panoptic_annotations, "images": images,
                              "categories": self.all_categories}

        coco_instance_json = {"info": info, "licenses": {}, "annotations": instance_annotations, "images": images,
                              "categories": self.all_categories}

        self.save_json(os.path.join(self.output_dir, self.dataset, "instances_" + self.dataset + ".json"),
                       coco_instance_json)
        self.save_json(os.path.join(self.output_dir, self.dataset, "panoptic_" + self.dataset + ".json"),
                       coco_panoptic_json)

    def process_keyframe(self, sd_record):
        pass

    def save_im(self, path, im):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, im)

    def get_all_categories(self):
        # get all the categories
        included_categories = set()
        coco_categories = []
        for category in self.nuim.category:
            parent_category = self.get_root_category(category['name'])
            if parent_category not in included_categories:
                included_categories.add(parent_category)

                coco_categories.append({
                    'id': len(coco_categories) + 1,
                    'name': parent_category,
                    'isthing': parent_category not in {'flat'},
                    'color': self.category_color_map[parent_category],
                    'supercategory': parent_category
                })
        return coco_categories

    def get_root_category(self, category):
        return category.split('.')[0]

    def create_panoptic_segment(self, mask, instance_id, im_id, category_id):
        mask_x, mask_y = np.where(mask > 0)
        if len(mask_x) == 0:
            return None
        x_min = int(np.min(mask_x))
        y_min = int(np.min(mask_y))

        x_max = int(np.max(mask_x))
        y_max = int(np.max(mask_y))

        x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

        if w == 0 or h == 0:
            return None

        segment_info = {"id": int(instance_id),
                        "category_id": int(category_id),
                        "area": w * h,
                        "iscrowd": 0,
                        "bbox": [x, y, w, h],
                        "image_id": im_id}
        return segment_info

    def save_json(self, instance_json_path, coco_instance_json):
        os.makedirs(os.path.dirname(instance_json_path), exist_ok=True)
        with open(instance_json_path, 'w') as f:
            json.dump(coco_instance_json, f, cls=utils.CustomJSONEncoder)


if __name__ == '__main__':
    # nuscenes_root = "/mnt/c/Users/eshrestha/Documents/nuimages"
    nuscenes_root = "C:\\Users\\eshrestha\\Documents\\nuimages"
    dataset = "v1.0-train"
    # output_dir = "/mnt/c/Users/eshrestha/Documents/nuimages/converted_coco"
    output_dir = "C:\\Users\\eshrestha\\Documents\\nuimages\\converted_coco"
    limit = 1000
    # converter = NuscenesToCocoPanopticConverter(nuscenes_root, dataset, output_dir, lim=limit)
    # converter.convert()

    root_path = os.path.join(f"{output_dir}_{limit}", dataset)
    coco_panoptic_json_path = os.path.join(root_path, "panoptic_v1.0-train.json")
    coco_instances_json_path = os.path.join(root_path, "instances_v1.0-train.json")
    coco_dataset_path = os.path.join(root_path, "images")
    coco_train_annotations_path = os.path.join(root_path, "panoptic_v1.0-train.json")
    coco_seg_path = os.path.join(root_path, "semantic")
    num_classes = 6

    dlseg = CocoPanopticSegmentation(num_classes=num_classes)
    register_dataset("test_dataset",
                     {
                               'stuff_classes': ['flat'],
                           },
                     coco_dataset_path,
                     coco_train_annotations_path,
                     coco_panoptic_json_path,
                     coco_seg_path,
                     coco_instances_json_path)
    utils.visualize_dataset_samples("test_dataset_separated")
