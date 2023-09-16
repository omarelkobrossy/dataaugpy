from enum import Enum


class BlurType(Enum):
    BOX_BLUR = 1
    SIMPLE_BLUR = 2
    GAUSSIAN_BLUR = 3
    MEDIAN_BLUR = 4

class NoiseType(Enum):
    GAUSSIAN_NOISE = 1
    FIXED_PATTERN_NOISE = 2
    BINDING_HORIZONTAL_NOISE = 3
    BINDING_VERTICAL_NOISE = 4
    BINDING_BOX_NOISE = 5

class WarpType(Enum):
    WARP_BULGE = 1

AVAILABLE_ANGLES = [90, 180, 270]
AVAILABLE_BLUR_TYPES = ("BOX_BLUR", "SIMPLE_BLUR", "GAUSSIAN_BLUR", "MEDIAN_BLUR")