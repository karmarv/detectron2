# Global Road Challenge

import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
from xml.etree import ElementTree
from xml.dom import minidom
import collections
import cv2

from fvcore.common.file_io import PathManager, file_lock
from fvcore.common.timer import Timer
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""



logger = logging.getLogger(__name__)

__all__ = ["load_images_ann_to_coco_dicts", "get_rdd_coco_instances_meta"]

# base_path = os.getcwd() + '/RoadDamageDataset/'
RDD_DAMAGE_CATEGORIES=[
        {"id": 1, "name": "D00", "color": [220, 20, 60] }, 
        {"id": 2, "name": "D01", "color": [165, 42, 42] }, 
        {"id": 3, "name": "D10", "color": [0, 0, 142]   }, 
        {"id": 4, "name": "D11", "color": [0, 0, 70]    }, 
        {"id": 5, "name": "D20", "color": [0, 60, 100]  }, 
        {"id": 6, "name": "D40", "color": [0, 80, 100]  }, 
        {"id": 7, "name": "D43", "color": [0, 0, 230]   }, 
        {"id": 8, "name": "D44", "color": [119, 11, 32] }, 
        {"id": 9, "name": "D50", "color": [128, 64, 128]},
        {"id": 10,"name": "D0w0","color": [96, 96, 96]  }
    ]

def load_image(img_path, image_file):
    img = cv2.imread(os.path.join(img_path, image_file))
    return img

def get_rdd_coco_instances_meta():
    thing_ids = [k["id"] for k in RDD_DAMAGE_CATEGORIES]
    thing_names = [k["name"] for k in RDD_DAMAGE_CATEGORIES]
    thing_colors = [k["color"] for k in RDD_DAMAGE_CATEGORIES]
    assert len(thing_ids) == 10, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in RDD_DAMAGE_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "thing_names" : thing_names
    }
    return ret

def load_images_ann_to_coco_dicts(basepath, splits_per_dataset):
    dataset_dicts = []

    metadata = get_rdd_coco_instances_meta()
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("[MoDet] Converting RDD dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for idx, regions_data in enumerate(splits_per_dataset):
        # Assume pre-defined datasets live in `./datasets`.
        ann_path = os.path.join(basepath, regions_data, "annotations/xmls")
        img_path = os.path.join(basepath, regions_data, "images")
        print("\tLoading ", idx,  " - ", img_path)
        # list annotations/xml dir and for each annotation load the data
        img_file_list = [filename for filename in os.listdir(img_path) if filename.endswith('.jpg')]
        for img_id, img_filename in enumerate(img_file_list):
            coco_image = {}
            coco_image["id"] = img_id+1
            coco_image["image_name"] = img_filename
            coco_image["file_name"] = os.path.join(img_path, img_filename)            

            ann_file = img_filename.split(".")[0] + ".xml"
            if os.path.isfile(os.path.join(ann_path, ann_file)):
                infile_xml = open(os.path.join(ann_path, ann_file))
                tree = ElementTree.parse(infile_xml)
                root = tree.getroot()
                for obj in root.iter('object'):
                    anno = {}
                    cls_name, xmlbox = obj.find('name').text, obj.find('bndbox')
                    xmin, xmax = float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text)
                    ymin, ymax = float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)
                    bbox = [xmin, ymin, xmax, ymax]  # (x0, y0, x1, y1)  -> (x0, y0, w, h) 
                    bbox = BoxMode.convert(bbox, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
                    bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()  # Computing areas using bounding boxes

                    #if img_id < 5: 
                    #    print(img_id, "\t Ann Class: ", cls_name, ", BBox", (xmin, xmax, ymin, ymax))
                    
                    # COCO requirement:
                    #   linking annotations to images
                    #   "id" field must start with 1
                    anno["id"] = len(coco_annotations) + 1
                    anno["image_id"] = coco_image["id"]
                    anno["bbox"] = [round(float(x), 3) for x in bbox]
                    anno["area"] = float(area)
                    anno["iscrowd"] = 0
                    anno["category_id"] = reverse_id_mapper(metadata.thing_names.index[cls_name])
                    anno["category_name"] = cls_name
                    anno["supercategory"] = regions_data
                    coco_annotations.append(anno)
                img_height = int(root.find('size').find('height').text)
                img_width  = int(root.find('size').find('width').text)
            else:
                im = cv2.imread(os.path.join(img_path, img_filename))
                img_height = im.shape[0]
                img_width  = im.shape[1]
            coco_image["height"] = img_height
            coco_image["width"] = img_width
            coco_images.append(coco_image)


    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated RDD Data COCO json file for Detectron2 .",
    }
    coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations

    return coco_dict  



def load_images_ann_dicts(basepath, splits_per_dataset):
    dataset_dicts = []
    thing_ids   = [k["id"] for k in RDD_DAMAGE_CATEGORIES]
    thing_names = [k["name"] for k in RDD_DAMAGE_CATEGORIES]

    for idx, regions_data in enumerate(splits_per_dataset):
        # Assume pre-defined datasets live in `./datasets`.
        ann_path = os.path.join(basepath, regions_data, "annotations/xmls")
        img_path = os.path.join(basepath, regions_data, "images")
        print("\tLoading ", idx,  " - ", img_path)
        # list annotations/xml dir and for each annotation load the data
        img_file_list = [filename for filename in os.listdir(img_path) if filename.endswith('.jpg')]
        for img_id, img_filename in enumerate(img_file_list):
            record = {}
            annos = []
            ann_file = img_filename.split(".")[0] + ".xml"
            if os.path.isfile(os.path.join(ann_path, ann_file)):
                infile_xml = open(os.path.join(ann_path, ann_file))
                tree = ElementTree.parse(infile_xml)
                root = tree.getroot()
                for obj in root.iter('object'):
                    anno = {}
                    cls_name, xmlbox = obj.find('name').text, obj.find('bndbox')
                    xmin, xmax = float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text)
                    ymin, ymax = float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)
                    #if img_id < 5: 
                    #    print(img_id, "\t Ann Class: ", cls_name, ", BBox", (xmin, xmax, ymin, ymax))
                    anno["category_id"]   = thing_ids[thing_names.index(cls_name)]
                    anno["category_name"] = cls_name
                    anno["bbox_mode"] = BoxMode.XYWH_ABS  # (x0, y0, w, h) 
                    bbox = [xmin, ymin, xmax, ymax]       # (x0, y0, x1, y1)  -> (x0, y0, w, h) 
                    bbox = BoxMode.convert(bbox, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) 
                    anno["bbox"] = bbox
                    annos.append(anno)
                img_height = int(root.find('size').find('height').text)
                img_width  = int(root.find('size').find('width').text)
            else:
                im = cv2.imread(os.path.join(img_path, img_filename))
                img_height = im.shape[0]
                img_width  = im.shape[1]
            record["image_id"] = img_id
            record["image_name"] = img_filename
            record["file_name"] = os.path.join(img_path, img_filename)
            record["height"] = img_height
            record["width"] = img_width
            record["annotations"] = annos
            record["supercategory"] = regions_data
            record["iscrowd"] = 0
            dataset_dicts.append(record)
    return dataset_dicts

if __name__ == "__main__":
    """
    Test the GRC xml dataset loader.

    Usage:
        env DETECTRON2_DATASETS=/media/rahul/Karmic/data python ./modet/data/datasets/rdd.py /media/rahul/Karmic/data 

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata

    import sys, os
    logger = setup_logger(name=__name__)

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    logger.info("loading dataset {}. \nAlready registered dataset: {} \n".format(_root, DatasetCatalog.list()))

    _PREDEFINED_SPLITS_RDD_MD = {}
    _PREDEFINED_SPLITS_RDD_MD["rdd"] = {
        "rdd2020_val"  : ( "RoadDamageDataset/val/Czech", 
                            "RoadDamageDataset/val/India", 
                            "RoadDamageDataset/val/Japan"),
        "rdd2020_train": ( "RoadDamageDataset/train/Czech", 
                            "RoadDamageDataset/train/India", 
                            "RoadDamageDataset/train/Japan")
    }
    
    meta = {"thing_classes": ["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44", "D50", "D0w0"]}
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_RDD_MD["rdd"].items():
        print(dataset_name, " - ", splits_per_dataset)
        inst_key = "{}".format(dataset_name)
        d = dataset_name.split("_")[1]
        load_images_ann_dicts(_root, splits_per_dataset)
        DatasetCatalog.register(
            inst_key,
            lambda d=d: load_images_ann_dicts(_root, splits_per_dataset),
        )
        MetadataCatalog.get(inst_key).set(evaluator_type="coco", **meta) 
    #dicts = load_coco_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(MetadataCatalog.get("rdd2020_train")))
    logger.info("Done loading {} samples.".format(MetadataCatalog.get("coco")))

