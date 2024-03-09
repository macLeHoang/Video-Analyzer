from dataclasses import dataclass
from typing import List, Tuple

DEFAULT_COLOR_PALETTE = [
    "A351FB",
    "FF4040",
    "FFA1A0",
    "FF7633",
    "FFB633",
    "D1D435",
    "4CFB12",
    "94CF1A",
    "40DE8A",
    "1B9640",
    "00D6C1",
    "2E9CAA",
    "00C4FF",
    "364797",
    "6675FF",
    "0019EF",
    "863AFF",
    "530087",
    "CD3AFF",
    "FF97CA",
    "FF39C9",
]

LEGACY_COLOR_PALETTE = [
    "#A351FB",
    "#E6194B",
    "#3CB44B",
    "#FFE119",
    "#0082C8",
    "#F58231",
    "#911EB4",
    "#46F0F0",
    "#F032E6",
    "#D2F53C",
    "#FABEBE",
    "#008080",
    "#E6BEFF",
    "#AA6E28",
    "#FFFAC8",
    "#800000",
    "#AAFFC3",
]

def _validate_color_hex(color_hex: str):
    color_hex = color_hex.lstrip("#")
    if not all(c in "0123456789abcdefABCDEF" for c in color_hex):
        raise ValueError("Invalid characters in color hash")
    if len(color_hex) not in (3, 6):
        raise ValueError("Invalid length of color hash")

@dataclass
class Color:
    """
    Represents a color in RGB format.

    This class provides methods to work with colors, including creating colors from hex
    codes, converting colors to hex strings, RGB tuples, and BGR tuples.

    Attributes:
        r (int): Red channel value (0-255).
        g (int): Green channel value (0-255).
        b (int): Blue channel value (0-255).

    | Constant   | Hex Code   | RGB              |
    |------------|------------|------------------|
    | `WHITE`    | `#FFFFFF`  | `(255, 255, 255)`|
    | `BLACK`    | `#000000`  | `(0, 0, 0)`      |
    | `RED`      | `#FF0000`  | `(255, 0, 0)`    |
    | `GREEN`    | `#00FF00`  | `(0, 255, 0)`    |
    | `BLUE`     | `#0000FF`  | `(0, 0, 255)`    |
    | `YELLOW`   | `#FFFF00`  | `(255, 255, 0)`  |
    | `ROBOFLOW` | `#A351FB`  | `(163, 81, 251)` |
    """

    r: int
    g: int
    b: int

    @classmethod
    def from_hex(cls, color_hex: str) -> Color:
        """Create a Color instance from a hex string.

        Parameters:
        -----------
            color_hex (str)
              Hex string of the color.
        
        Returns:
        -----------
            Color
              Instance representing the color.
        """
        _validate_color_hex(color_hex)
        color_hex = color_hex.lstrip("#")
        if len(color_hex) == 3:
            color_hex = "".join(c * 2 for c in color_hex)
        r, g, b = (int(color_hex[i : i + 2], 16) for i in range(0, 6, 2))
        return cls(r, g, b)
    
    def as_rgb(self) -> Tuple[int, int, int]:
        return self.r, self.g, self.b

    def as_bgr(self) -> Tuple[int, int, int]:
        return self.b, self.g, self.r
    
    @classmethod
    def WHITE(cls):
        return Color.from_hex("#FFFFFF")

    @classmethod
    def BLACK(cls):
        return Color.from_hex("#000000")

    @classmethod
    def RED(cls):
        return Color.from_hex("#FF0000")

    @classmethod
    def GREEN(cls):
        return Color.from_hex("#00FF00")

    @classmethod
    def BLUE(cls):
        return Color.from_hex("#0000FF")

    @classmethod
    def YELLOW(cls):
        return Color.from_hex("#FFFF00")


@dataclass
class ColorPalette:
    colors: List[Color]

    @classmethod
    def DEFAULT(cls) -> ColorPalette:
        return ColorPalette.from_hex(color_hex_list=DEFAULT_COLOR_PALETTE)

    @classmethod
    def LEGACY(cls) -> ColorPalette:
        return ColorPalette.from_hex(color_hex_list=LEGACY_COLOR_PALETTE)

    @classmethod
    def from_hex(cls, color_hex_list: List[str]) -> ColorPalette:
        colors = [Color.from_hex(color_hex) for color_hex in color_hex_list]
        return cls(colors)

    def by_idx(self, idx: int) -> Color:
        if idx < 0:
            raise ValueError("idx argument should not be negative")
        idx = idx % len(self.colors)
        return self.colors[idx]
