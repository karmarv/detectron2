import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from copy import deepcopy

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from deeplab import add_deeplab_config, build_lr_scheduler


# Environment variable setup
DETECTRON2_DATASETS = "/media/rahul/Karmic/data/"
ROADDAMAGE_DATASET  = os.path.join(DETECTRON2_DATASETS, "rdd2020/")
DATASET_BASE_PATH   = ROADDAMAGE_DATASET
# constants
WINDOW_NAME = "Semantic Segmentations detections"

# Configuration setup
cfg = get_cfg()
add_deeplab_config(cfg)
cfg.merge_from_file("./configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml")
cfg.OUTPUT_DIR            = "./output/run_90k/"
cfg.MODEL.WEIGHTS         = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.DEVICE          = "cuda"
cfg.DATASETS.TEST         = ("cityscapes_fine_sem_seg_test", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
cpu_device = torch.device("cpu")
predictor = DefaultPredictor(cfg)


def cv2_imshow(im, time_out=10000):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 1024)
    cv2.imshow(WINDOW_NAME, im.get_image()[:, :, ::-1])
    if cv2.waitKey(time_out) == 27:
        cv2.destroyAllWindows() # esc to quit
        print("Closing the view")


def predict_visualize(full_image_path):
    print("Image:  ", full_image_path)
    im = cv2.imread(full_image_path)
    predictions = predictor(im)
    # Convert image from OpenCV BGR format to Matplotlib RGB format.
    image = im[:, :, ::-1]
    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
    if "sem_seg" in predictions:
        print("Keys :  ", len(predictions["sem_seg"]))
        vis_output = visualizer.draw_sem_seg(
            predictions["sem_seg"].argmax(dim=0).to(cpu_device)
        )
    if "instances" in predictions:
        instances = predictions["instances"].to(cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
    return predictions, vis_output

print("Metadata List: ", MetadataCatalog.list())
#image_filepath = os.path.join(ROADDAMAGE_DATASET, "train/Japan/images", "Japan_000000.jpg")    # single image
image_filepath = os.path.join(ROADDAMAGE_DATASET, "train/India/images")                         # directory
if os.path.isdir(image_filepath):
    try:
        for id, imfile in enumerate(glob.glob(os.path.join(image_filepath, '*.jpg'))): # assuming jpg images
            print(id, ".) \t Loading Images: ", imfile)
            predictions, vis_output = predict_visualize(imfile)
            cv2_imshow(vis_output)
    except Exception as e:
        raise Exception('Error loading data from %s: %s\n' % (image_filepath, e))
else:
    predictions, vis_output = predict_visualize(image_filepath)
    cv2_imshow(vis_output)