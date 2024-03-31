import sys
sys.path.insert(1, ".")

from omegaconf import OmegaConf
import numpy as np
import cv2 

from VideoAnalyzer.apis import Analyzer
from VideoAnalyzer.annotators.core import BoundingBoxAnnotator, LabelAnnotator
from VideoAnalyzer.utils import get_pylogger
logger = get_pylogger()

if __name__ == "__main__":
    path = "video/test1.mp4"
    video = cv2.VideoCapture(path)

    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Resized_Window", 1200, 650) 

    analyzer = Analyzer(config="configs/default.yaml")
    box_ann = BoundingBoxAnnotator()
    label_ann = LabelAnnotator()

    while True:
        suc, frame = video.read()
        if not suc:
            break

        result = analyzer.do_detect(frame)[0]
        result = analyzer.do_track(metadata=result)
 
        f = box_ann.annotate(frame, result)
        f = label_ann.annotate(frame, result)

        cv2.imshow("Resized_Window", f)
        if cv2.waitKey(0) & 0xff == 27:
            break

    video.release()
    cv2.destroyAllWindows()
    



    