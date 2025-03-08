import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
from config import *

class Camera:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.parameters: CameraParameters
        self._load_parameters()

    def _load_parameters(self) -> None:
        with self.file_path.open() as file:
            camera_data = json.load(file)
        
        self.parameters = self.extract_parameters_from_json(camera_data)

    @staticmethod
    def extract_parameters_from_json(camera_data: Dict[str, Any]) -> CameraParameters:
        return CameraParameters(
            intrinsic=IntrinsicParameters(
                matrix=np.array(camera_data[JsonParameters.Intrisic][JsonParameters.Doubles], dtype=np.float64).reshape(3, 3)
            ),
            extrinsic=ExtrinsicParameters(
                rotation_matrix=np.array(
                    camera_data[JsonParameters.Extrinsic][JsonParameters.Tf][JsonParameters.Doubles], dtype=np.float64
                    ).reshape(4, 4)[:3, :3],
                translation_vector=np.array(
                    camera_data[JsonParameters.Extrinsic][JsonParameters.Tf][JsonParameters.Doubles], dtype=np.float64
                    ).reshape(4, 4)[:3, 3].reshape(3, 1)
            ),
            distortion=DistortionParameters(
                coefficients=np.array(camera_data[JsonParameters.Distorcion][JsonParameters.Doubles], dtype=np.float64)
            ),
            resolution=(
                int(camera_data[JsonParameters.Resolution][JsonParameters.Width]),
                int(camera_data[JsonParameters.Resolution][JsonParameters.Height])
            )
        )

    def get_parameters(self) -> CameraParameters:
        return self.parameters

    @staticmethod
    def load_cameras(config_dir: str, num_cameras: int = 4) -> List[CameraParameters]:
        config_path = Path(config_dir)
        return [Camera(config_path / f"{i}.json").get_parameters() for i in range(num_cameras)]
