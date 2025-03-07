import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.use("TkAgg")
parameters =  aruco.DetectorParameters()
dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) #N sei se esse é o correto porém detecta
arucoDetector = aruco.ArucoDetector(dictionary, parameters)
def read_videos():
    for i in range(4):
        file_name = f"videos/camera-0{i}.mp4"
        vid = cv2.VideoCapture(file_name)

        while True:
            _, img = vid.read()
            if img is None:
                print("Empty Frame")
                break
            corners, ids, rejectedImgPoints = arucoDetector.detectMarkers(img)
            frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
            #cv2.imshow('output', frame_markers)
            
            
            
            #if cv2.waitKey(1) == ord('q'):
                #break

        #cv2.destroyAllWindows()
    return corners, ids # aqui deve retornar alguma coisa para estimação da matriz essencial
