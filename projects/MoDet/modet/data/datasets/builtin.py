# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog

from modet.data.datasets.builtin_meta import _get_builtin_metadata
from modet.data.datasets.cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from modet.data.datasets.register_coco import register_coco_instances, register_coco_panoptic_separated
from modet.data.datasets.rdd import load_images_ann_dicts

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO_MD = {}
_PREDEFINED_SPLITS_COCO_MD["coco_md"] = {

    "coco_md_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_md_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_md_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_md_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_md_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO_MD["coco_md_person"] = {
    "keypoints_coco_md_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_md_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_md_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_MD_PANOPTIC = {
    "coco_md_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_md_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_md_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco_md(root):
    logger = logging.getLogger(__name__)
    logger.info("[MoDet] Register in COCO format from {}".format(root))
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO_MD.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            print(key, image_root, json_file)
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_MD_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_md_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )


# ==== Predefined splits for raw cityscapes images ===========


_RAW_CITYSCAPES_MD_SPLITS = {
    "cityscapes_md_fine_{task}_train": ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train"),
    "cityscapes_md_fine_{task}_val": ("cityscapes/leftImg8bit/val", "cityscapes/gtFine/val"),
    "cityscapes_md_fine_{task}_test": ("cityscapes/leftImg8bit/test", "cityscapes/gtFine/test"),
}

_RAW_CITYSCAPES_MD_PANOPTIC_SPLITS = {
    "cityscapes_md_panoptic_fine_{task}_train": ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train"),
    "cityscapes_md_panoptic_fine_{task}_val": ("cityscapes/leftImg8bit/val", "cityscapes/gtFine/val"),
    "cityscapes_md_panoptic_fine_{task}_test": ("cityscapes/leftImg8bit/test", "cityscapes/gtFine/test"),
}

def register_all_cityscapes(root):
    # Regular cityscapes dataset
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_MD_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes_md")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_sem_seg", **meta
        )
    
    # Panoptic cityscapes dataset [TODO: evaluator and config verification ]
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_MD_PANOPTIC_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes_md_panoptic")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_sem_seg", **meta
        )



# ==== Predefined splits for raw grc images ===========

_PREDEFINED_SPLITS_RDD_MD = {}
_PREDEFINED_SPLITS_RDD_MD["rdd"] = {
    "rdd2020_val"  : ( "RoadDamageDataset/val/Czech", 
                       "RoadDamageDataset/val/India", 
                       "RoadDamageDataset/val/Japan"),
    "rdd2020_train": ( "RoadDamageDataset/train/Czech", 
                       "RoadDamageDataset/train/India", 
                       "RoadDamageDataset/train/Japan")
}

def register_all_rdd_datasets(root):
    logger = logging.getLogger(__name__)
    logger.info("[MoDet] Register GRC in COCO format from {}".format(root))
    meta = _get_builtin_metadata("rdd")
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_RDD_MD["rdd"].items():
        inst_key = "{}".format(dataset_name)
        d = dataset_name.split("_")[2]
        print(dataset_name, "\t", splits_per_dataset)
        #load_images_ann_dicts(_root, dataset_name, splits_per_dataset)
        DatasetCatalog.register(
            inst_key,
            lambda d=d: load_images_ann_dicts(root, splits_per_dataset),
        )
        MetadataCatalog.get(inst_key).set(evaluator_type="coco", **meta) 
    return None


# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_md(_root)
register_all_cityscapes(_root)
register_all_rdd_datasets(_root)