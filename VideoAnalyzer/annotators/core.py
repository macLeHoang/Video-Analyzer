from .base import BaseAnnotator, MetaDatas
from .draw.color import ColorPalette, Color
from .draw.position import Position
from .utils import ColorLookup, resolve_color

from typing import Union, Optional, Tuple
import cv2
import numpy as np
    
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
                 metadatas):
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
        for data_idx in range(len(metadatas)):
            x1, y1, x2, y2 = metadatas.xyxy[data_idx].astype(int)
            color = resolve_color(
                color=self.color,
                metadatas=metadatas,
                data_idx=data_idx,
                color_lookup=self.color_lookup
            )
            cv2.rectangle(scene, (x1, y1), (x2, y2), color=color.as_bgr(), thickness=self.thickness)
        return scene


class LabelAnnotator:
    """
    A class for annotating labels on an image using provided detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT(),
        text_color: Color = Color.WHITE(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        text_position: Position = Position.TOP_LEFT,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        border_radius: int = 0,
        CLASS_NAME_DATA_FIELD=""
    ):
        """
        Parameters:
        -----------
            color (Union[Color, ColorPalette]): 
                The color or color palette to use for annotating the text background.
            text_color (Color): The color to use for the text.
            text_scale (float): Font scale for the text.
            text_thickness (int): Thickness of the text characters.
            text_padding (int): Padding around the text within its background box.
            text_position (Position): Position of the text relative to the detection.
                Possible values are defined in the `Position` enum.
            color_lookup (str): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            border_radius (int): The radius to apply round edges. If the selected
                value is higher than the lower dimension, width or height, is clipped.
        """
        self.border_radius: int = border_radius
        self.color: Union[Color, ColorPalette] = color
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.text_anchor: Position = text_position
        self.color_lookup: ColorLookup = color_lookup

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates: Tuple[int, int],
        text_wh: Tuple[int, int],
        position: Position,
    ) -> Tuple[int, int, int, int]:
        
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh

        if position == Position.TOP_LEFT:
            return center_x, center_y - text_h, center_x + text_w, center_y
        elif position == Position.TOP_RIGHT:
            return center_x - text_w, center_y - text_h, center_x, center_y
        elif position == Position.TOP_CENTER:
            return (
                center_x - text_w // 2,
                center_y - text_h,
                center_x + text_w // 2,
                center_y,
            )
        elif position == Position.CENTER or position == Position.CENTER_OF_MASS:
            return (
                center_x - text_w // 2,
                center_y - text_h // 2,
                center_x + text_w // 2,
                center_y + text_h // 2,
            )
        elif position == Position.BOTTOM_LEFT:
            return center_x, center_y, center_x + text_w, center_y + text_h
        elif position == Position.BOTTOM_RIGHT:
            return center_x - text_w, center_y, center_x, center_y + text_h
        elif position == Position.BOTTOM_CENTER:
            return (
                center_x - text_w // 2,
                center_y,
                center_x + text_w // 2,
                center_y + text_h,
            )
        elif position == Position.CENTER_LEFT:
            return (
                center_x - text_w,
                center_y - text_h // 2,
                center_x,
                center_y + text_h // 2,
            )
        elif position == Position.CENTER_RIGHT:
            return (
                center_x,
                center_y - text_h // 2,
                center_x + text_w,
                center_y + text_h // 2,
            )

    @staticmethod
    def draw_rounded_rectangle(
        scene: np.ndarray,
        xyxy: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        border_radius: int,
    ) -> np.ndarray:
        x1, y1, x2, y2 = xyxy
        width = x2 - x1
        height = y2 - y1

        border_radius = min(border_radius, min(width, height) // 2)

        rectangle_coordinates = [
            ((x1 + border_radius, y1), (x2 - border_radius, y2)),
            ((x1, y1 + border_radius), (x2, y2 - border_radius)),
        ]
        circle_centers = [
            (x1 + border_radius, y1 + border_radius),
            (x2 - border_radius, y1 + border_radius),
            (x1 + border_radius, y2 - border_radius),
            (x2 - border_radius, y2 - border_radius),
        ]

        for coordinates in rectangle_coordinates:
            cv2.rectangle(
                img=scene,
                pt1=coordinates[0],
                pt2=coordinates[1],
                color=color,
                thickness=-1,
            )
        for center in circle_centers:
            cv2.circle(
                img=scene,
                center=center,
                radius=border_radius,
                color=color,
                thickness=-1,
            )
        return scene    
    
    def annotate(self, scene, metadatas, labels=None):
        if metadatas.xyxy.size == 0:
            return scene
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        anchors_coordinates = metadatas.get_anchors_coordinates(
            anchor=self.text_anchor
        ).astype(int)

        for data_idx, center_coordinates in enumerate(anchors_coordinates):
            color = resolve_color(
                color=self.color,
                metadatas=metadatas,
                data_idx=data_idx,
                color_lookup=self.color_lookup
            )

            if labels is not None:
                text = labels[data_idx]
            elif metadatas.class_id is not None:
                text = str(metadatas.class_id[data_idx])
                if metadatas.track_id is not None:
                    text += f"- {metadatas.track_id[data_idx]}"
            else:
                text = str(metadatas)

            text_w, text_h = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]
            text_w_padded = text_w + 2 * self.text_padding
            text_h_padded = text_h + 2 * self.text_padding
            text_background_xyxy = self.resolve_text_background_xyxy(
                center_coordinates=tuple(center_coordinates),
                text_wh=(text_w_padded, text_h_padded),
                position=self.text_anchor,
            )

            text_x = text_background_xyxy[0] + self.text_padding
            text_y = text_background_xyxy[1] + self.text_padding + text_h

            self.draw_rounded_rectangle(
                scene=scene,
                xyxy=text_background_xyxy,
                color=color.as_bgr(),
                border_radius=self.border_radius,
            )

            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=self.text_color.as_rgb(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )

        return scene
