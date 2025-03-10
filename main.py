import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from src.calibration.parameters import Camera
from src.calibration.matrices import MatrixProcessor
from src.video_process.plot_results import plot_3d_points
from src.video_process.process import compute_camera_matrices, process_video_for_camera, reconstruct_3d_points
from src.parser import Parser

videos_prefix_name: str = "camera-0"

def main():
    parser = Parser().get_arguments()
    config_dir = parser.config_dir
    videos_path = parser.videos_path

    cameras = Camera.load_cameras(config_dir)

    file_list = [f"{videos_path}/{videos_prefix_name}{i}.mp4" for i in range(4)]

    with Pool(processes=4) as pool:
        video_marker_positions = pool.map(process_video_for_camera, file_list)

    camera_projection_matrices = compute_camera_matrices(cameras)

    matrices = []
    for i in range(len(video_marker_positions[0])):
        frame_matrices = [np.array(video_marker_positions[j][i]) for j in range(len(file_list))]
        processor = MatrixProcessor(frame_matrices, camera_projection_matrices)
        concatenated_matrix = processor.process_and_concatenate()
        matrices.append(concatenated_matrix)

    resulting_A = reconstruct_3d_points(matrices)
    plot_3d_points(resulting_A)

if __name__ == "__main__":
    main()
