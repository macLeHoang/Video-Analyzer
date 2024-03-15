from .base import BaseAnnotator, MetaDatas
from .draw.color import ColorPalette, Color
from .utils import ColorLookup, resolve_color

from typing import Union, Optional
import cv2
import numpy as np

class Annotator(BaseAnnotator):
    def __init__(self) -> None:
        super().__init__()
    
class BoundingBoxAnnotator(BaseAnnotator):
    """
    A class for drawing bounding boxes on an image using provided detections.
    """

    def __init__(self,
                 color: Union[Color, ColorPalette] = ColorPalette.DEFAULT(),
                 thickness: int = 2,
                 color_lookup: ColorLookup = ColorLookup.CLASS):
        """
        Parameters:
        -----------
            color (Union[Color, ColorPalette]): 
                The color or color palette to use for annotating detections.
            thickness (int): 
                Thickness of the bounding box lines.
            color_lookup (str): 
                Strategy for mapping colors to annotations. Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.color_lookup: ColorLookup = color_lookup

    def annotate(self, 
                 scene, 
                 metadatas: MetaDatas,
                 custom_color_lookup: Optional[np.ndarray] = None):
        """Annotates the given scene with bounding boxes based on the provided detections.

        Parameters:
        -----------
            scene, ndarray: 
                The image where bounding boxes will be drawn. 
            meatadatas, MetaDatas:
                Object detections to annotate.
            custom_color_lookup, Optional[np.ndarray]:
                Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
        -----------
            The annotated image, matching the type of `scene`
        """
        if custom_color_lookup is not None:
            color_lookup = custom_color_lookup
        else:
            color_lookup = self.color_lookup

        for data_idx in range(len(metadatas)):
            x1, y1, x2, y2 = metadatas.xyxy[data_idx].astype(int)
            color = resolve_color(
                color=self.color,
                metadatas=metadatas,
                data_idx=data_idx,
                color_lookup=color_lookup
            )
            cv2.rectangle(scene, (x1, y1), (x2, y2), color=color.as_bgr(), thickness=self.thickness)
        return scene
