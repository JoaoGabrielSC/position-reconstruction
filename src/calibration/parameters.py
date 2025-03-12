import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
from src.config import *

class Camera:
    """
    A class representing a camera configuration and its parameters.

    This class loads the camera parameters from a JSON file, extracts relevant intrinsic, extrinsic, distortion, and resolution data, 
    and provides methods to access these parameters
    """

    def __init__(self, file_path: Path):
        """
        Initializes a Camera object by loading camera parameters from a given JSON file.

        Args:
            file_path (Path): The file path to the JSON file containing the camera configuration.
        """
        self.file_path = Path(file_path)
        self.parameters: CameraParameters
        self._load_parameters()

    def _load_parameters(self) -> None:
        """
        Loads the camera parameters from the JSON file and stores them in the `parameters` attribute.

        This method parses the JSON file and extracts the intrinsic, extrinsic, distortion, and resolution data,
        then stores them as a `CameraParameters` object.

        Returns:
            None
        """
        with self.file_path.open() as file:
            camera_data = json.load(file)
        
        self.parameters = self.extract_parameters_from_json(camera_data)

    @staticmethod
    def extract_parameters_from_json(camera_data: Dict[str, Any]) -> CameraParameters:
        """
        Extracts camera parameters from a JSON-like dictionary.

        Args:
            camera_data (Dict[str, Any]): A dictionary representing the camera configuration, including intrinsic, 
                                          extrinsic, distortion, and resolution data.

        Returns:
            CameraParameters: A `CameraParameters` object containing the extracted camera parameters.
        """
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
        """
        Returns the loaded camera parameters.

        Returns:
            CameraParameters: The camera parameters extracted from the JSON file.
        """
        return self.parameters

    @staticmethod
    def load_cameras(config_dir: str, num_cameras: int = 4) -> List[CameraParameters]:
        """
        Loads multiple cameras from the given configuration directory.

        Args:
            config_dir (str): The directory containing the camera configuration JSON files.
            num_cameras (int, optional): The number of camera files to load (default is 4).

        Returns:
            List[CameraParameters]: A list of `CameraParameters` objects for the specified number of cameras.
        """
        config_path = Path(config_dir)
        return [Camera(config_path / f"{i}.json").get_parameters() for i in range(num_cameras)]
