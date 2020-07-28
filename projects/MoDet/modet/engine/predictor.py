import logging
import time
import numpy as np
import atexit
from collections import deque

import torch
from detectron2.engine.defaults import DefaultPredictor

logger = logging.getLogger(__name__)

class MoDetPredictor(DefaultPredictor):

    def format_response(self, original_image, predictions, cpu_device = "cpu"):
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = original_image[:, :, ::-1]
        logger.info("Preds: {}".format(predictions.keys()))
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            
        else:
            if "sem_seg" in predictions:
                sem_segs = predictions["sem_seg"].argmax(dim=0).to(cpu_device)

            if "instances" in predictions:
                instances = predictions["instances"].to(cpu_device)
        
        return predictions

    """
        Pred Class
    """
    def pred_format(self, original_image, cpu_device = "cpu"):
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
            start_time = time.time()
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            logger.info(
                "{}: {} in {:.2f}s".format(
                    "Image",
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            return self.format_response(original_image, predictions, cpu_device)
        return None

