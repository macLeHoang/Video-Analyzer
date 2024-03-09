from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from .utils import get_data_item

@dataclass
class Metadata: 
    """
    The `Detections` allows you to convert results from a variety of object detection
    and segmentation models into a single, unified format.
    """
    xyxy: np.ndarray = None
    mask: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None
    data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict)

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)
    
    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[float],
            Optional[int],
            Optional[int],
            Dict[str, Union[np.ndarray, List]],
        ]
    ]:
        """
        Iterates over the Detections object and yield a tuple of
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


ImageType = Union[np.ndarray, Image.Image]
class BaseAnnotator(ABC):
    """
    TODO: ImageType = TypeVar("ImageType", np.ndarray, Image.Image)

    An image of type `np.ndarray` or `PIL.Image.Image`.

    Unlike a `Union`, ensures the type remains consistent. If a function
    takes an `ImageType` argument and returns an `ImageType`, when you
    pass an `np.ndarray`, you will get an `np.ndarray` back.
    """
    @abstractmethod
    def annotate(self, scene: ImageType, detections: Metadata) -> np.ndarray:
        pass


