import random
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import glob
import os

def callImage():
    # Charger l'image
    image = Image.open('flower.jpg')

    # Afficher l'image
    plt.imshow(image)
    plt.axis('off')  # Masquer les axes
    plt.show()