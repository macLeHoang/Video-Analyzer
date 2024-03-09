import sys
sys.path.insert(1, ".")

from omegaconf import OmegaConf
import yaml

from VideoAnalyzer.detection.core import Detectors
from VideoAnalyzer.utils import get_pylogger
logger = get_pylogger()

if __name__ == "__main__":
    detector = Detectors(config="configs/default.yaml")

    import numpy as np
    input = np.ones((640, 640, 3), dtype=np.uint8)

    detector.do_detect([input])
    