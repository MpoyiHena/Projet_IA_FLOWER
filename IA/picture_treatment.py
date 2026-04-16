import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
import os

def callImage():
    # Chemin vers le dossier contenant les images
    path = r'C:\Dossier E\Henallux\M1\Systeme inteligent\Projet\flowers\Dataset_flowers\Training Data'
    
    # Trouver toutes les images .jpeg dans les sous-dossiers
    images = glob.glob(os.path.join(path, '**', '*.jpeg'), recursive=True)
    
    # Afficher chaque image avec un délai de 2 secondes
    for img_path in images:
        img = cv2.imread(img_path)
        cv2.imshow('Image', img)
        key = cv2.waitKey(20)  # Attendre 2 secondes
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()