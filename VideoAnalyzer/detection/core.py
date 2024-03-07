import torch
from ultralytics import YOLO
from omegaconf import DictConfig

from .utils import model_zoo
from VideoAnalyzer.utils import get_pylogger
logger = get_pylogger()

class Detections:
    """
    The `Detections` allows you to convert results from a variety of object detection
    and segmentation models into a single, unified format.
    """
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.load_configurations()
    
    def load_configurations(self):
        model = self.config.Detection.model
        weight = self.config.Detection.weight

        if model == "yolov8":
            self.model = YOLO(weight)
            self.kwargs_ = {
                "imgsz"  : self.config.Detection.imgsz,
                "conf"   : self.config.Detection.conf,
                "classes": self.config.Detection.classes,
                "verbose": self.config.Detection.verbose,
                "device" : self.config.Detection.device,
            }
        elif model == "yolov5":
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight)
            self.kwargs_ = {
                "imgsz"  : self.config.Detection.imgsz,
                "conf"   : self.config.Detection.conf,
                "classes": self.config.Detection.classes,
                "verbose": self.config.Detection.verbose,
                "device" : self.config.Detection.device,
            }
        else:
            logger.warning(f"Model type {model} is not supported !!!")
            raise
        self.do_detect_ = model_zoo[model]

    def do_detect(self, batch):
        results = self.do_detect_(self.model, batch, **self.kwargs_)
        return results