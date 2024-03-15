from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image

from .utils import get_data_item

@dataclass
class MetaDatas: 
    """
    The `MetaDatas` allows you to convert results from a variety of object detection
    and segmentation models into a single, unified format.
    """
    xyxy: np.ndarray = None
    mask: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None
    data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict)

    def __len__(self):
        """Returns the number of data in the MetaDatas object.
        """
        return len(self.xyxy)
    
    def __iter__(self):
        """Iterates over the Detections object and yield a tuple of
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
    
    def __repr__(self) -> str:
        pass


ImageType = Union[np.ndarray, Image.Image]

class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, scene: ImageType, detections: MetaDatas) -> np.ndarray:
        raise NotImplementedError


