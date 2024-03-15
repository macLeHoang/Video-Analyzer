import sys
sys.path.insert(1, ".")

from omegaconf import OmegaConf
import cv2 

from VideoAnalyzer.apis.core import Analyzer
from VideoAnalyzer.annotators.core import BoundingBoxAnnotator
from VideoAnalyzer.utils import get_pylogger
logger = get_pylogger()

if __name__ == "__main__":
    

    path = "video/test1.mp4"
    video = cv2.VideoCapture(path)

    analyzer = Analyzer(config="configs/default.yaml")

    _, frame = video.read()

    bbannot = BoundingBoxAnnotator()
    r = analyzer.do_detect(frame)
    f = bbannot.annotate(frame, r[0])

    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Resized_Window", 1200, 650) 
    cv2.imshow("Resized_Window", f)
    cv2.waitKey(0)

    