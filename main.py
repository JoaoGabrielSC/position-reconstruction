import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from multiprocessing import Pool
from src.calibration.parameters import Camera
from src.calibration.matrices import MatrixProcessor
from src.video_process.plot_results import plot_3d_points
from src.video_process.process import VideoProcessor
from src.parser import Parser

videos_prefix_name: str = "camera-0"
THREAD_POOL_SIZE: int = 4

def main():
    video_processor = VideoProcessor(adjustment_value=0.05)
    parser = Parser().get_arguments()
    config_dir = parser.config_dir
    videos_path = parser.videos_path

    cameras = Camera.load_cameras(config_dir)

    file_list = [f"{videos_path}/{videos_prefix_name}{i}.mp4" for i in range(4)]

    print("Processando os 4 vídeos ao mesmo tempo com Paralelismo Multithread")
    with Pool(processes=4) as pool:
        video_marker_positions = pool.map(video_processor.process_video_for_camera, file_list)
    
    camera_projection_matrices = video_processor.compute_camera_matrices(cameras)

    print(camera_projection_matrices)

    matrices = []
    for i in range(len(video_marker_positions[0])):
        frame_matrices = [np.array(video_marker_positions[j][i]) for j in range(len(file_list))]
        processor = MatrixProcessor(frame_matrices, camera_projection_matrices)
        concatenated_matrix = processor.process_and_concatenate()
        matrices.append(concatenated_matrix)

    resulting_A = video_processor.reconstruct_3d_points(matrices)
    plot_3d_points(resulting_A)

if __name__ == "__main__":
    main()
