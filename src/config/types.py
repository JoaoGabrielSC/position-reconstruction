from dataclasses import dataclass
from enum import StrEnum
from typing import Tuple
import numpy as np

class JsonParameters(StrEnum):
    Intrisic = "intrinsic",
    Extrinsic = "extrinsic"
    Doubles = "doubles",
    Tf = "tf",
    Resolution = "resolution",
    Distorcion = "distortion"
    Width = "width",
    Height = "height",

@dataclass
class IntrinsicParameters:
    matrix: np.ndarray

@dataclass
class ExtrinsicParameters:
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray

@dataclass
class DistortionParameters:
    coefficients: np.ndarray

@dataclass
class CameraParameters:
    intrinsic: IntrinsicParameters
    extrinsic: ExtrinsicParameters
    distortion: DistortionParameters
    resolution: Tuple[int, int]
