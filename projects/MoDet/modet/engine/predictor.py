import logging
import numpy as np
import atexit
from collections import deque

import torch
from detectron2.engine.defaults import DefaultPredictor


class MoDetPredictor(DefaultPredictor):
    """
        Pred Class
    """
    def pred(self, original_image):
        """
            Prediction handler
            Args:
                original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

            Returns:
                predictions (dict):
                    the output of the model for one image only.
                    See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
        return None

