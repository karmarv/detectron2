import logging
import numpy as np
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class MoDetVisualizer(Visualizer):
    """
        Visualizer Class
    """
    def visualize(self) -> None:
        """
           Visualize
        """
        return None

class MoDetVideoVisualizer(VideoVisualizer):
    def visualize(self) -> None:
        """
           Visualize
        """        
        return None
