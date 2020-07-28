import logging
import numpy as np
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2

from detectron2.data import MetadataCatalog

logger = logging.getLogger(__name__)

class MetaView(object):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
       

    def form_meta(self, dataset):
        """
        Args:
            dataset_name
        Retun:
            A formatted metadata to be used for visualization
        """
        metav = self.metadata.as_dict()

        print("Annotation Keys: {}".format(metav))
        print("Annotation Json: {}".format(metav["json_file"]))
        print("Thing: {} \n {}".format(len(metav["thing_classes"]), metav["thing_classes"]))
        print("Stuff: {} \n {}".format(len(metav["stuff_classes"]), metav["stuff_classes"]))

        return metav