from cv2.aruco import Dictionary
import cv2
import numpy as np

from src.calibration.matrices import MatrixProcessor
from src.config.types import CameraParameters

class VideoProcessor:
    def __init__(self) -> None:
        pass

    def process_video_for_camera(self, file_name: str) -> list:
        """
        Processa um vídeo específico para detectar marcadores ArUco e extrair as posições dos marcadores.
        
        Args:
            file_name (str): O caminho do arquivo de vídeo a ser processado.
        
        Returns:
            list: Lista contendo as posições dos marcadores para cada frame do vídeo processado.
                  Caso não haja marcadores, retorna uma lista vazia.
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()

        return self.process_videos([file_name], aruco_dict, parameters)[0]
    
    def process_videos(self, file_list: list[str], aruco_dict: Dictionary, parameters):
        """
        Processa vídeos para detectar marcadores ArUco e extrair posições.
        
        Args:
            file_list (list of str): Lista de caminhos para arquivos de vídeo.
            aruco_dict: Dicionário ArUco para detecção.
            parameters: Parâmetros do detector ArUco.
        
        Returns:
            list: Lista de posições dos marcadores para cada frame e câmera.
        """
        video_marker_positions = []
        
        for file in (file_list):
            vid = cv2.VideoCapture(file)
            current_frame_marker_positions = []
            file_idx = file.split('camera-')[1].split('.mp4')[0]
            idx = int(file_idx[-1])
            window_name = f"Video {idx}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Definindo a posição da janela na tela para ver todos os videos
            x = (idx % 2) * 640
            y = (idx // 2) * 480
            cv2.moveWindow(window_name, x, y)

            while True:
                ret, img = vid.read()
                if not ret or img is None:
                    break

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, _ = detector.detectMarkers(gray)
                corners_with_id_0 = [corners[i] for i in range(len(ids)) if ids[i] == 0] if ids is not None else []

                processor = MatrixProcessor(corners_with_id_0)
                marker_positions = processor.calculate_means() if corners_with_id_0 else np.array([[]])
                current_frame_marker_positions.append(marker_positions)

                resized_img = cv2.resize(img, (640, 480))

                if ids is not None:
                    for corner in corners_with_id_0:
                        resized_corners = corner * (640 / img.shape[1], 480 / img.shape[0])
                        resized_corners = resized_corners.astype(int)
                        cv2.aruco.drawDetectedMarkers(resized_img, [resized_corners], np.array([0]))

                cv2.imshow(window_name, resized_img)
                if cv2.waitKey(1) == ord('q'):
                    break
            idx +=1

            cv2.destroyWindow(window_name)
            video_marker_positions.append(current_frame_marker_positions)
        
        return video_marker_positions

    @staticmethod
    def compute_camera_matrices(cameras: list[CameraParameters]):
        """
        Calcula as matrizes de projeção das câmeras.
        
        Args:
            cameras (list): Lista de objetos de câmera carregados.
        
        Returns:
            list: Lista das matrizes de projeção das câmeras.
        """
        projection_matrices = []
        
        for camera in cameras:
            K = camera.intrinsic.matrix
            R = camera.extrinsic.rotation_matrix
            T = camera.extrinsic.translation_vector
            Rcamtw = R.T
            tcamtw = -R.T @ T
            RT_camtw = np.hstack((Rcamtw, tcamtw))
            proj_m = np.concatenate((RT_camtw, [[0, 0, 0, 1]]), axis=0)
            P_t = K @ np.eye(3, 4) @ proj_m
            projection_matrices.append(P_t)
        
        return projection_matrices

    @staticmethod
    def reconstruct_3d_points(matrices: list[np.ndarray]):
        """
        Reconstrói pontos 3D a partir das matrizes processadas.
        
        Args:
            matrices (list of ndarray): Lista de matrizes processadas.
        
        Returns:
            list: Lista de pontos 3D reconstruídos.
        """
        resulting_A = []
        for matrix in matrices:
            U, D, Vt = np.linalg.svd(matrix)
            resulting_A.append(Vt[-1, :4])
        return resulting_A
