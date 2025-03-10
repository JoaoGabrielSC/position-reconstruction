# Descrição do Trabalho
Neste trabalho devemos detectar o robô nos vídeos das 4 câmeras do espaço inteligente e obter a reconstrução da sua posição 3D no mundo. Feito isso, vocês deverão gerar um gráfico da posição do robô, mostrando a trajetória que ele realizou.

Para detectar o robô será usado um marcador ARUCO acoplado a sua plataforma. Rotinas de detecção desse tipo de marcador poderão ser usadas para obter sua posição central, assim como as suas quinas nas imagens. Essas informações, juntamente com os dados de calibração das câmeras, poderão ser usadas para localização 3D do robô.

Informações a serem consideradas:

- Só é necessário a reconstrução do ponto central do robô (ou suas quinas, se vocês acharem melhor). Para isso, vocês podem usar o método explicado no artigo fornecido como material adicional ou nos slides que discutimos em sala de aula.

- O robô está identificado por um marcador do tipo ARUCO - Código ID 0 (zero) - Tamanho 30 x 30 cm

- Os vídeos estão sincronizados para garantir que, a cada quadro, vocês estarão processando imagens do robô capturadas no mesmo instante.

- A calibração das câmeras é fornecida em 4 arquivos no formato JSON (Junto com os arquivos JSON estou fornecendo uma rotina para leitura e extração dos dados de calibração).

- Rotinas de detecção dos marcadores Aruco em imagens e vídeo são fornecidas para ajudar no desenvolvimento do trabalho.

ATENÇÃO: Existem rotinas de detecção de ARUCO que já fornecem sua localização e orientação 3D, se a calibração da câmera e o tamanho do padrão forem fornecidas. Essas rotinas poderão ser usadas para fazer comparações com a reconstrução 3D fornecida pelo trabalho de vocês, mas não serão aceitas como o trabalho a ser feito. Portanto, lembrem-se que vocês deverão desenvolver a rotina de reconstrução, a partir da detecção do ARUCO acoplado ao robô nas imagens 2D capturadas nos vídeos.


DATA DE ENTREGA: 17/03/2023

# Estrutua das pastas
    .
    ├── Makefile
    ├── README.md
    ├── main.py
    ├── requirements.txt
    ├── src
    │   ├── calibration
    │   │   ├── 0.json
    │   │   ├── 1.json
    │   │   ├── 2.json
    │   │   ├── 3.json
    │   │   ├── matrices.py
    │   │   └── parameters.py
    │   ├── config
    │   │   ├── __init__.py
    │   │   └── types.py
    │   ├── parser
    │   │   └── __init__.py
    │   └── video_process
    │       ├── plot_results.py
    │       └── process.py
    └── videos
        ├── camera-00.mp4
        ├── camera-01.mp4
        ├── camera-02.mp4
        ├── camera-03.mp4
        └── video_aruco.py

# Rode o Projeto

O projeto pode ser executado das seguintes formas:

```bash
make run
# ou
python main.py
# ou
python main.py --config_dir "caminho/pasta/calibration" --videos_path "caminho/para/videos"
```
Se o primeiro ou o segundo método for utilizado, sem nenhuma flag, os arquivos serão buscados nos caminhos padrão definidos nas opções acima.
Portanto, certifique-se de que os arquivos .json de calibração e os vídeos .mp4 estejam na pasta correta antes da execução.
