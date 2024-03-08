import torch
from ultralytics import YOLO
from omegaconf import OmegaConf
import yaml
from dataclasses import dataclass, field, InitVar

import numpy as np
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .utils import model_zoo, get_data_item
from VideoAnalyzer.utils import get_pylogger
logger = get_pylogger()

@dataclass
class Detections:
    """
    The `Detections` allows you to convert results from a variety of object detection
    and segmentation models into a single, unified format.
    """
    xyxy: np.ndarray = None
    mask: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None
    data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict)
    config: InitVar[str] = None
    
    def __post_init__(self, config) -> None:
        if config is not None:
            self.load_configurations(config)
    
    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)
    
    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[float],
            Optional[int],
            Optional[int],
            Dict[str, Union[np.ndarray, List]],
        ]
    ]:
        """
        Iterates over the Detections object and yield a tuple of
        `(xyxy, mask, confidence, class_id, tracker_id, data)` for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.mask[i] if self.mask is not None else None,
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
                get_data_item(self.data, i)
            )

    def load_configurations(self, config):
        with open(config, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.warning(exc)
                raise      

        model = self.config["Detection"]["model"]
        weight = self.config["Detection"]["weight"]

        if model == "yolov8":
            self.model = YOLO(weight)
            self.kwargs_ = {
                "imgsz"  : self.config["Detection"]["imgsz"],
                "conf"   : self.config["Detection"]["conf"],
                "classes": self.config["Detection"]["classes"],
                "verbose": self.config["Detection"]["verbose"],
                "device" : self.config["Detection"]["device"],
            }
        elif model == "yolov5":
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight)
            self.kwargs_ = {
                "imgsz"  : self.config["Detection"]["imgsz"],
                "conf"   : self.config["Detection"]["conf"],
                "classes": self.config["Detection"]["classes"],
                "verbose": self.config["Detection"]["verbose"],
                "device" : self.config["Detection"]["device"],
            }
        else:
            logger.warning(f"Model type {model} is not supported !!!")
            raise

        self.__detect = model_zoo[model]

    def do_detect(self, batch):
        det_objects_ = []
        dets = self.__detect(self.model, batch, **self.kwargs_)

        for det in dets:
            det_objects_.append(Detections(xyxy=det))
        return det_objects_

