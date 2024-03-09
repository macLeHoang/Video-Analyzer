import torch
from ultralytics import YOLO
from omegaconf import OmegaConf
import yaml
from typing import Any, List

from VideoAnalyzer.annotators.base import Metadata
from .utils import model_zoo
from VideoAnalyzer.utils import get_pylogger
logger = get_pylogger()

class Detectors:
    def __init__(self, config) -> None:
        self.load_configurations(config)

    def load_configurations(self, config: str) -> None:
        with open(config, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(exc)
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

    def do_detect(self, batch: Any) -> List[Metadata]:
        det_objects_ = []
        dets = self.__detect(self.model, batch, **self.kwargs_)

        for det in dets:
            det_objects_.append(Metadata(xyxy=det))
        return det_objects_

