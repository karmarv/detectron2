import logging
import numpy as np
import atexit
from collections import deque

from detectron2.engine.defaults import DefaultPredictor


class MoDetPredictor(DefaultPredictor):
    """
        Pred Class
    """
    def pred(self) -> None:
        """
           Pred
        """
        return None

