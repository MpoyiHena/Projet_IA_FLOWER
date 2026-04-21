import random
import time
from pathlib import Path
import cv2
import numpy as np

# Importer les fonctions de picture_treatment.py
from picture_treatment import get_image_folder, choose_random_image

def detect_flowers(image_path: Path):
    """
    Détecte les fleurs dans l'image en utilisant une segmentation couleur simple
    et dessine des rectangles autour des régions détectées, avec le nom de la fleur
    et le pourcentage de ressemblance.
    """
    # Charger l'image avec cv2
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Impossible de charger l'image : {image_path}")
        return None

    # Ajouter une bordure en haut pour l'espace texte (50 pixels)
    image = cv2.copyMakeBorder(image, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Extraire le nom de la fleur depuis le nom du fichier (ex: Rose_test_85.jpeg -> Rose)
    flower_name = image_path.stem.split('_')[0]
    confidence = 40  # Pourcentage fictif, à remplacer par une vraie classification

    # Convertir en HSV pour une meilleure segmentation couleur
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir les plages de couleur pour les fleurs (exemple : rouges, jaunes, roses)
    # Ajuster selon les couleurs des fleurs dans le dataset
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([160, 255, 255])

    # Masques pour chaque couleur
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

    # Combiner les masques
    mask = mask_red1 | mask_red2 | mask_yellow | mask_pink

    # Appliquer des opérations morphologiques pour nettoyer le masque
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Trouver les contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner des rectangles autour des contours suffisamment grands
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Seuil pour éviter les petits artefacts
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Ajouter le nom de la fleur et le pourcentage en haut à gauche du rectangle
            text = f"{flower_name}__{confidence}%"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image

def show_image_cv2(image, delay: float = 1.0):
    """
    Affiche l'image avec cv2.
    """
    cv2.imshow('Image avec fleurs encadrées', image)
    cv2.waitKey(int(delay * 1000))
    cv2.destroyAllWindows()

def track_images_with_detection(num_images: int = 5, inter_delay: float = 0.5, display_delay: float = 4.0):
    """
    Fonction pour suivre/afficher une séquence d'images avec détection des fleurs.
    inter_delay : temps d'attente entre deux fleurs (secondes)
    display_delay : durée d'affichage de chaque image (secondes)
    """
    folder = get_image_folder()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {folder}")

    print(f"Début du suivi d'images avec détection dans : {folder}")
    for i in range(num_images):
        image_path = choose_random_image(folder)
        print(f"Traitement de l'image {i+1} : {image_path.name}")
        detected_image = detect_flowers(image_path)
        if detected_image is not None:
            show_image_cv2(detected_image, display_delay)
        time.sleep(inter_delay)

if __name__ == "__main__":
    # Exemple d'utilisation : afficher 5 images avec détection
    track_images_with_detection(num_images=10, inter_delay=0.5, display_delay=4.0)