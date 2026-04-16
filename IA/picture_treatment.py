import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def callImage():
    # Charger l'image
    image = Image.open('flower.jpg')

    # Afficher l'image
    plt.imshow(image)
    plt.axis('off')  # Masquer les axes
    plt.show()