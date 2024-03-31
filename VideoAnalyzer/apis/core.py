from omegaconf import DictConfig, OmegaConf
from typing import Union
import os
import numpy as np
import cv2

from VideoAnalyzer.annotators import MetaDatas
from VideoAnalyzer.detection.core import Detectors
from VideoAnalyzer.track.core import Trackers
from VideoAnalyzer.utils import get_pylogger
logger = get_pylogger()

class Analyzer:
    def __init__(self, config: Union[DictConfig, str] = "configs/default.yaml"):
        self.cfg = self.load_config(config)
        self.configure_modules()
    

    def load_config(self, config):
        if isinstance(config, str) and config.endswith(("yaml", "yml")):
            if not os.path.exists(config):
                logger.warning(f"Path is not exist !!!. \
                               Using default config instead.")
                config = "configs/default.yaml"
            cfg = OmegaConf.load(config)
        elif isinstance(config, DictConfig):
            cfg = config
        else:
            logger.error(f"Expecting config is <{DictConfig.__name__}> instance or path to yaml file !!!")
            raise TypeError

        return cfg


    def configure_modules(self):
        self.supported_mode = []

        if "annotation" in self.cfg:
            self.verbose_action = {
                "save" : self.cfg.annotation.save,
                "show" : self.cfg.annotation.show,
                "print": self.cfg.annotation.print
            }
            self.supported_mode.append("annotation")

        if "detection" in self.cfg:
            logger.info(f"Initiating <{self.cfg.detection.model}> detection module")
            self.detector = Detectors(self.cfg.detection)
            self.supported_mode.append("detection")
        
        if "track" in self.cfg:
            logger.info(f"Initiating <{self.cfg.track.model}> track module")
            self.tracker = Trackers(self.cfg.track)
            self.supported_mode.append("track")


    def do_detect(self, scene):
        if not ("detection" in self.supported_mode):
            logger.warning(f"Detection config is not found !!!. \
                           Please add the detection field to config to continue...")
            return MetaDatas(xyxy=np.array([]))
        
        det_result = self.detector.do_detect(scene) # list of metadata
        return det_result


    def do_track(self, metadata=None, scene=None):
        if not ("track" in self.supported_mode):
            logger.warning(f"Detection config is not found !!!. \
                           Please add the track field to config to continue...")
            return metadata

        if metadata is None:
            assert scene is not None, f"Expected scene input"
            metadata = self.do_detect(scene)[0]
        
        tracklet = self.tracker.do_track(metadata) # metadata
        return tracklet
    

    def do_track_video(self, video=None, metadata_list=None):
        if metadata_list is None:
            assert video is not None, f"Expected video input"
            if isinstance(video, str):
                if not os.path.exists(video):
                    logger.error(f"Video is not exists !!!")
                    raise
                cap = cv2.VideoCapture(video)
            elif isinstance(video, cv2.VideoCapture):
                cap = video
            
            logger.info("Start tracking. Press Esc to stop!")
            while True:
                suc, frame = cap.read()
                if not suc:
                    logger.info(f"Done processing.")
                    break
                self.do_track(scene=frame)
        else:
            for metadata in metadata_list:
                self.do_track(metadata=metadata)


    def verbose(self, scene=None, metadata=None):
        if self.verbose_action["print"]:
            logger.info(metadata)
        elif self.verbose_action["show"]:
            pass # TODO: do_show
        elif self.verbose_action["save"]:
            pass # TODO: do_save