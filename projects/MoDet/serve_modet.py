# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
from PIL import Image
from flask import Flask, jsonify, request

import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

from modet.engine.predictor import MoDetPredictor


app = Flask(__name__)
logger = setup_logger(output="output/serve_modet.log", name="serve.modet")

@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        imfile = request.files['file']
        if imfile is not None:
            image = Image.open(imfile)  
            image = convert_PIL_to_numpy(image, format="BGR")
            print("Image:", image.shape)
            prediction_idx = run_on_image(image)
            print(prediction_idx)
            #class_id, class_name = render_prediction(prediction_idx)
            #return jsonify({'class_id': class_id, 'class_name': class_name})

def run_on_image(image):
    """
    Args:
        image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            This is the format used by OpenCV.

    Returns:
        predictions (dict): the output of the model.
    """
    start_time = time.time()
    predictions = predictor.pred_format(image, cpu_device = cpu_device)
    logger.info(
        "{}: {} in {:.2f}s".format(
            "Format + Predict Image",
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )
    return predictions

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device on which to run the detection model.",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output for visualizations.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


""" 
    - Run Server
        python serve_modet.py --config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x_md.yaml --opts MODEL.WEIGHTS ./output/model_final.pth
    - Test Client
        curl -X POST -H "Content-Type: multipart/form-data" http://0.0.0.0:5000/predict -F "file=@input1.jpg"
"""
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger.info("Device: " + str(args.device) + ", Arguments: " + str(args))

    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    cpu_device = torch.device("cpu")
    predictor = MoDetPredictor(cfg)
    app.run()

