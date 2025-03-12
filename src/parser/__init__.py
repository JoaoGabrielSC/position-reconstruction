from argparse import Namespace, ArgumentParser

class Parser:
    def __init__(self):
        self.description = self.set_description()
        self.parser = ArgumentParser(description=self.description)

        self.parser.add_argument("--config_dir", required=False, type=str, default="src/calibration",
                                 help="Caminho para pasta contendo os JSON de calibração")
        self.parser.add_argument("--videos_path", required=False, type=str, default="./videos",
                                 help="Caminho para a pasta dos vídeos")

    def set_description(self):
        return "Processamento de vídeos para detecção de marcadores ArUco e reconstrução 3D."

    def get_arguments(self) -> Namespace:
        return self.parser.parse_args()
