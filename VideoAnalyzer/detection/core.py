import torch
from ultralytics import YOLO
import yaml
from typing import Any, List, Union
from omegaconf import OmegaConf, DictConfig

from VideoAnalyzer.annotators.base import MetaDatas
from .utils import detector_zoo

from VideoAnalyzer.utils import get_pylogger
logger = get_pylogger()

class Detectors:
    def __init__(self, config) -> None:
        self.load_configurations(config)

    def load_configurations(self, config: Union[str, DictConfig]) -> None:
        if isinstance(config, str) and str.endswith(("yaml", "yml")):
            with open(config, 'r') as stream:
                try:
                    config = yaml.safe_load(stream)
                    self.cfg = config["detection"]
                except yaml.YAMLError as exc:
                    logger.error(exc)
                    raise
        elif isinstance(config, DictConfig):
            self.cfg = OmegaConf.to_object(config)

        model = self.cfg["model"]
        weight = self.cfg["weight"]

        if model == "yolov8":
            self.model = YOLO(weight)
            self.kwargs_ = {
                "imgsz"  : self.cfg["imgsz"],
                "conf"   : self.cfg["conf"],
                "classes": self.cfg["classes"],
                "verbose": self.cfg["verbose"],
                "device" : self.cfg["device"],
            }
        elif model == "yolov5":
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight)
            self.kwargs_ = {
                "imgsz"  : self.cfg["imgsz"],
                "conf"   : self.cfg["conf"],
                "classes": self.cfg["classes"],
                "verbose": self.cfg["verbose"],
                "device" : self.cfg["device"],
            }
        else:
            logger.warning(f"Model type {model} is not supported !!!")
            raise

        self.__detect = detector_zoo[model]

    
    def do_detect(self, batch: Any) -> List[MetaDatas]:
        batch = self._validate_batch(batch)
        det_objects_ = []
        dets = self.__detect(self.model, batch, **self.kwargs_)

        for det in dets:
            det_objects_.append(MetaDatas(xyxy=det[:, :4], 
                                          class_id=det[:, -1].astype(int),
                                          confidence=det[:, -2]))
        return det_objects_
    

    def _validate_batch(self, batch: Any):
        if self.model in ["yolov5", "yolov8"]:
            if not isinstance(batch, list):
                batch = list(batch)
        return batch


