import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #type: ignore

def plot_3d_points(resulting_A: list[np.ndarray]) -> None:
    """
    Plota pontos 3D a partir de uma lista de vetores homogeneizados.
    
    Args:
        resulting_A (list of ndarray): Lista de pontos 3D reconstruídos.
    """
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    
    for Vt in resulting_A:
        x, y, z, w = Vt[0], Vt[1], Vt[2], Vt[3]
        ax.scatter(x/w, y/w, z/w, c='r', marker='o')
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()
