import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from typing import Union

from VideoAnalyzer.detection.core import Detectors
from VideoAnalyzer.utils import get_pylogger
logger = get_pylogger()

class Analyzer(nn.Module):
    def __init__(self, config: Union[DictConfig, str]):
        super().__init__()
        self.cfg = self.load_config(config)
        self.configure_modules()
    

    def load_config(self, config):
        if isinstance(config, str) and config.endswith(("yaml", "yml")):
            cfg = OmegaConf.load(config)
        elif isinstance(config, DictConfig):
            cfg = config
        else:
            logger.warning(f"Expect config is <DictConfig> instance or path to yaml file !!!")
            raise

        return cfg


    def configure_modules(self):
        self.supported_mode = []

        if "annotation" in self.cfg and self.cfg.annotation.active:
            self.verbose_action = {
                "save" : self.cfg.annotation.save,
                "show" : self.cfg.annotation.show,
                "print": self.cfg.annotation.print
            }

        if "detection" in self.cfg:
            self.detector = Detectors(self.cfg.detection)
            self.supported_mode.append("detection")


    def do_detect(self, scene):
        det_result = self.detector.do_detect(scene)
        return det_result
    

    def verbose(self, scene=None, metadata=None):
        if self.verbose_action["print"]:
            logger.info(metadata)
        elif self.verbose_action["show"]:
            pass # TODO: do_show
        elif self.verbose_action["save"]:
            pass # TODO: do_save