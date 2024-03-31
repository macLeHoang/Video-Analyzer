import numpy as np
from typing import Dict, Union, List
from enum import Enum

from .draw.color import Color, ColorPalette

def get_data_item(
    data: Dict[str, Union[np.ndarray, List]],
    index: Union[int, slice, List[int], np.ndarray],
) -> Dict[str, Union[np.ndarray, List]]:
    """Retrieve a subset of the data dictionary based on the given index.

    Parameters:
    -----------
        data, dict:
            The data dictionary of the Detections object.
        index, int or slice, list, ndarray
            The index or indices specifying the subset to retrieve.

    Returns:
    -----------
        A subset of the data dictionary corresponding to the specified index.
    """
    subset_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            subset_data[key] = value[index]

        elif isinstance(value, list):
            if isinstance(index, slice):
                subset_data[key] = value[index]
            elif isinstance(index, (list, np.ndarray)):
                subset_data[key] = [value[i] for i in index]
            elif isinstance(index, int):
                subset_data[key] = [value[index]]
            else:
                raise TypeError(f"Unsupported index type: {type(index)}")
            
        else:
            raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")

    return subset_data


class ColorLookup(Enum):
    """
    Enumeration class to define strategies for mapping colors to annotations.

    This enum supports three different lookup strategies:
        - `INDEX`: Colors are determined by the index of the detection within the scene.
        - `CLASS`: Colors are determined by the class label of the detected object.
        - `TRACK`: Colors are determined by the tracking identifier of the object.
    """

    INDEX = "index"
    CLASS = "class"
    TRACK = "track"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def resolve_color_idx(
    meatadatas,
    data_idx: int,
    color_lookup: Union[ColorLookup, np.ndarray] = ColorLookup.CLASS,
) -> int:
    if data_idx >= len(meatadatas):
        raise ValueError(
            f"Detection index {data_idx}"
            f"is out of bounds for detections of length {len(meatadatas)}"
        )

    if isinstance(color_lookup, np.ndarray):
        if len(color_lookup) != len(meatadatas):
            raise ValueError(
                f"Length of color lookup {len(color_lookup)}"
                f"does not match length of detections {len(meatadatas)}"
            )
        return color_lookup[data_idx]
    
    elif color_lookup == ColorLookup.INDEX:
        return data_idx
    
    elif color_lookup == ColorLookup.CLASS:
        if meatadatas.class_id is None:
            raise ValueError(
                "Could not resolve color by class because"
                "Detections do not have class_id"
            )
        return meatadatas.class_id[data_idx]
    
    elif color_lookup == ColorLookup.TRACK:
        if meatadatas.tracker_id is None:
            raise ValueError(
                "Could not resolve color by track because"
                "Detections do not have tracker_id"
            )
        return meatadatas.tracker_id[data_idx]


def get_color_by_index(color: Union[Color, ColorPalette], idx: int) -> Color:
    if isinstance(color, ColorPalette):
        return color.by_idx(idx)
    return color


def resolve_color(
    color: Union[Color, ColorPalette],
    metadatas,
    data_idx: int,
    color_lookup: Union[ColorLookup, np.ndarray] = ColorLookup.CLASS,
) -> Color:
    idx = resolve_color_idx(
        meatadatas=metadatas,
        data_idx=data_idx,
        color_lookup=color_lookup,
    )
    return get_color_by_index(color=color, idx=idx)

def calculate_masks_centroids(masks: np.ndarray) -> np.ndarray:
    """
    Calculate the centroids of binary masks in a tensor.

    Parameters:
        masks (np.ndarray): A 3D NumPy array of shape (num_masks, height, width).
            Each 2D array in the tensor represents a binary mask.

    Returns:
        A 2D NumPy array of shape (num_masks, 2), where each row contains the x and y
            coordinates (in that order) of the centroid of the corresponding mask.
    """
    num_masks, height, width = masks.shape
    total_pixels = masks.sum(axis=(1, 2))

    # offset for 1-based indexing
    vertical_indices, horizontal_indices = np.indices((height, width)) + 0.5
    # avoid division by zero for empty masks
    total_pixels[total_pixels == 0] = 1

    def sum_over_mask(indices: np.ndarray, axis: tuple) -> np.ndarray:
        return np.tensordot(masks, indices, axes=axis)

    aggregation_axis = ([1, 2], [0, 1])
    centroid_x = sum_over_mask(horizontal_indices, aggregation_axis) / total_pixels
    centroid_y = sum_over_mask(vertical_indices, aggregation_axis) / total_pixels

    return np.column_stack((centroid_x, centroid_y)).astype(int)