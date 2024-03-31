from omegaconf import OmegaConf, DictConfig
from typing import Any, List

from third_parties.byte_track.byte_tracker import BYTETracker
from VideoAnalyzer.annotators.base import MetaDatas
from VideoAnalyzer.utils import get_pylogger
logger = get_pylogger()

from .utils.model_zoo import tracker_zoo

class Trackers:
    def __init__(self, config) -> None:
        self.load_configurations(config)

    def load_configurations(self, config: DictConfig):
        self.cfg = config

        model = self.cfg.model
        if model == "byte_track":
            self.tracker = BYTETracker(self.cfg.kwargs.args, self.cfg.kwargs.frame_rate)
            self.kwargs_ = {}
        else:
            logger.error(f"Model type {model} is not supported !!!")
            raise

        self.__track = tracker_zoo[model]
    
    def do_track(self, metadata: MetaDatas) -> MetaDatas:
        tracklets = self.__track(self.tracker, metadata, **self.kwargs_) # xyxy, score, track_id
        xyxy, score, cls_id, track_id = tracklets
        
        return MetaDatas(xyxy=xyxy,
                         confidence=score,
                         class_id=cls_id,
                         track_id=track_id)