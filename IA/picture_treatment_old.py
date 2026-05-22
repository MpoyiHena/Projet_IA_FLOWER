# ============================================================================
# TRAITEMENT D'IMAGES - Preprocessing et augmentation pour classification
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV pour le traitement d'images
import keras
from keras import layers
from PIL import Image, ImageEnhance  # Pour les ajustements avancés
import os
import random
import glob
from pathlib import Path

# ============================================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================================

def charger_donnees(chemin_train, chemin_test, taille_image=256, batch_size_train=15746, batch_size_test=2460):
    """
    Charge les datasets d'entraînement et de test depuis les répertoires

    Paramètres:
        chemin_train (str): Chemin vers le répertoire d'entraînement
        chemin_test (str): Chemin vers le répertoire de test
        taille_image (int): Taille de redimensionnement (256x256 par défaut)
        batch_size_train (int): Nombre d'images à traiter par lot (entraînement)
        batch_size_test (int): Nombre d'images à traiter par lot (test)

    Retour:
        train, test: Datasets chargés et préprocessés
        class_names: Liste des noms des classes
    """
    # Charge le dataset d'entraînement
    train = keras.utils.image_dataset_from_directory(
        chemin_train,
        labels="inferred",  # Les labels viennent des noms des sous-dossiers
        batch_size=batch_size_train,
        image_size=(taille_image, taille_image),  # Redimensionne à 256x256
        format="tf"  # Format TensorFlow
    )

    # Charge le dataset de test
    test = keras.utils.image_dataset_from_directory(
        chemin_test,
        labels="inferred",
        batch_size=batch_size_test,
        image_size=(taille_image, taille_image),
        format="tf"
    )

    # Récupère les noms des classes (types de fleurs)
    class_names = train.class_names

    return train, test, class_names


# ============================================================================
# 2. VISUALISATION DES IMAGES
# ============================================================================

def afficher_apercu_images(dataset, class_names, nb_images=9):
    """
    Affiche un aperçu de N images du dataset

    Paramètres:
        dataset: Dataset d'images à afficher
        class_names (list): Liste des noms des classes
        nb_images (int): Nombre d'images à afficher (défaut: 9 en grille 3x3)
    """
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):  # Prend le premier lot
        for i in range(min(nb_images, len(images))):
            # Crée une sous-figure dans une grille 3x3
            ax = plt.subplot(3, 3, i + 1)
            # Affiche l'image en couleur 8-bit
            plt.imshow(np.array(images[i]).astype("uint8"))
            # Ajoute le label de la classe
            plt.title(class_names[int(labels[i])])
            # Masque les axes
            plt.axis("off")
    plt.show()


# ============================================================================
# 3. DATA AUGMENTATION - Augmente la variété des données d'entraînement
# ============================================================================

# Définit les couches d'augmentation de données
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),  # Retournement horizontal aléatoire
    layers.RandomRotation(0.1),  # Rotation aléatoire jusqu'à 10%
    layers.RandomZoom(0.1),  # Zoom aléatoire (optionnel)
]

def augmentation_donnees(images):
    """
    Applique les transformations d'augmentation de données aux images
    Cela aide le modèle à être plus robuste et à mieux généraliser

    Paramètres:
        images: Batch d'images à augmenter

    Retour:
        images: Images transformées
    """
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


# ============================================================================
# 4. NORMALISATION - Ramène les valeurs entre 0 et 1
# ============================================================================

def normaliser_donnees(x_train, x_test):
    """
    Normalise les données pour que les valeurs de pixels soient entre 0 et 1
    La normalisation aide le modèle à converger plus rapidement

    Paramètres:
        x_train: Données d'entraînement
        x_test: Données de test

    Retour:
        x_train, x_test: Données normalisées
    """
    print('Before normalization : Min={}, max={}'.format(x_train.min(), x_train.max()))

    # Récupère la valeur maximale (généralement 255 pour les images)
    xmax = x_train.max()

    # Normalise en divisant par la valeur max
    x_train = x_train / xmax
    x_test = x_test / xmax

    print('After normalization  : Min={}, max={}'.format(x_train.min(), x_train.max()))

    return x_train, x_test


# ============================================================================
# 5. TRAITEMENTS D'IMAGES AVANCES
# ============================================================================

def ajuster_contraste(image, facteur=1.5):
    """
    Augmente ou diminue le contraste d'une image
    Le contraste définit la différence entre les zones claires et sombres

    Paramètres:
        image: Image PIL ou numpy array
        facteur (float): Facteur de contraste (1.0 = original, >1.0 = plus de contraste)

    Retour:
        image: Image avec contraste ajusté
    """
    # Convertit numpy array en image PIL si nécessaire
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))

    # Ajuste le contraste
    enhancer = ImageEnhance.Contrast(image)
    image_contrastee = enhancer.enhance(facteur)

    return np.array(image_contrastee)


def ajuster_luminosite(image, facteur=1.2):
    """
    Augmente ou diminue la luminosité d'une image
    Plus de luminosité = image plus claire

    Paramètres:
        image: Image PIL ou numpy array
        facteur (float): Facteur de luminosité (1.0 = original, >1.0 = plus clair)

    Retour:
        image: Image avec luminosité ajustée
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))

    enhancer = ImageEnhance.Brightness(image)
    image_lumineuse = enhancer.enhance(facteur)

    return np.array(image_lumineuse)


def appliquer_flou_gaussien(image, kernel_size=5):
    """
    Applique un flou gaussien à l'image
    Utile pour réduire le bruit et les détails inutiles

    Paramètres:
        image: Image numpy array
        kernel_size (int): Taille du kernel (5, 7, 9, etc... doit être impair)

    Retour:
        image: Image floutée
    """
    # Assurez-vous que la taille du kernel est impaire
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Applique le flou gaussien
    image_floue = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return image_floue


def detecter_contours(image, method='canny'):
    """
    Détecte les contours d'une image
    Utile pour identifier les frontières des objets et structures importantes

    Paramètres:
        image: Image numpy array (en niveaux de gris de préférence)
        method (str): 'canny' ou 'sobel'

    Retour:
        image: Image avec contours détectés (en niveaux de gris)
    """
    # Convertit en niveaux de gris si nécessaire
    if len(image.shape) == 3:
        image_grise = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_grise = image

    if method == 'canny':
        # Détection de contours avec Canny (très populaire)
        # Les seuils 100 et 200 peuvent être ajustés
        contours = cv2.Canny(image_grise, 100, 200)

    elif method == 'sobel':
        # Détection avec Sobel (gradient - calcule la dérivée)
        sobelx = cv2.Sobel(image_grise, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image_grise, cv2.CV_64F, 0, 1, ksize=5)
        contours = np.sqrt(sobelx**2 + sobely**2).astype('uint8')

    return contours


def convertir_espace_couleur(image, espace='HSV'):
    """
    Convertit l'image d'un espace de couleur à un autre
    Différents espaces de couleur révèlent différentes informations

    Paramètres:
        image: Image numpy array en RGB
        espace (str): 'HSV', 'LAB', 'GRAY', 'YCrCb'

    Retour:
        image: Image convertie dans le nouvel espace de couleur
    """
    if espace == 'HSV':
        # Teinte (Hue), Saturation, Valeur - utile pour identifier les couleurs
        # Sépare la couleur de la luminance
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    elif espace == 'LAB':
        # L (luminance), a et b (couleurs) - sépare bien luminance/couleur
        # Très utile pour la segmentation basée sur les couleurs
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    elif espace == 'GRAY':
        # Niveaux de gris - perd l'information de couleur
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    elif espace == 'YCrCb':
        # Utilisé en vidéo/compression JPEG
        # Y = luminance, Cr/Cb = chrominance (couleur)
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    return image


def egaliser_histogramme(image):
    """
    Égalise l'histogramme de l'image en niveaux de gris
    Améliore le contraste en distribuant mieux les valeurs de pixels
    Utile pour les images sous-exposées ou sur-exposées

    Paramètres:
        image: Image numpy array en niveaux de gris

    Retour:
        image: Image avec histogramme égalisé
    """
    # Convertit en niveaux de gris si nécessaire
    if len(image.shape) == 3:
        image_grise = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_grise = image

    # Applique l'égalisation d'histogramme adaptatif (CLAHE)
    # CLAHE = Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_egalisee = clahe.apply(image_grise)

    return image_egalisee


def egaliser_histogramme_couleur(image):
    """
    Égalise l'histogramme pour chaque canal de couleur
    Améliore le contraste global tout en préservant les couleurs

    Paramètres:
        image: Image numpy array en couleur (RGB)

    Retour:
        image: Image avec histogramme égalisé par canal
    """
    # Convertit de RGB à HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Applique l'égalisation sur le canal V (valeur/luminance)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_hsv[:, :, 2] = clahe.apply(image_hsv[:, :, 2])

    # Reconvertit en RGB
    image_egalisee = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    return image_egalisee


def appliquer_filtre_median(image, kernel_size=5):
    """
    Applique un filtre médian pour réduire le bruit
    Efficace pour éliminer le bruit "poivre et sel" (valeurs aberrantes)
    Préserve mieux les contours que le flou gaussien

    Paramètres:
        image: Image numpy array
        kernel_size (int): Taille du kernel (doit être impair)

    Retour:
        image: Image filtrée
    """
    return cv2.medianBlur(image, kernel_size)


def redimensionner_image(image, largeur, hauteur):
    """
    Redimensionne une image à une taille spécifique

    Paramètres:
        image: Image numpy array
        largeur (int): Largeur cible
        hauteur (int): Hauteur cible

    Retour:
        image: Image redimensionnée
    """
    return cv2.resize(image, (largeur, hauteur))


# ============================================================================
# 6. PIPELINE COMPLET DE PREPROCESSING
# ============================================================================

def pipeline_preprocessing_complet(image, normaliser=True, augmenter=False):
    """
    Applique un pipeline complet de preprocessing à une image

    Paramètres:
        image: Image numpy array
        normaliser (bool): Appliquer la normalisation
        augmenter (bool): Appliquer l'augmentation de données

    Retour:
        image: Image traitée
    """
    # 1. Égalisation d'histogramme (améliore le contraste)
    image = egaliser_histogramme_couleur(image)

    # 2. Réduction du bruit avec filtre médian
    image = appliquer_filtre_median(image, kernel_size=5)

    # 3. Ajustement du contraste
    image = ajuster_contraste(image, facteur=1.2)

    # 4. Normalisation
    if normaliser:
        image = image.astype(np.float32) / 255.0

    # 5. Augmentation (optionnel)
    if augmenter:
        image = augmentation_donnees(image)

    return image


# ============================================================================
# 7. FONCTIONS D'AFFICHAGE (du code original)
# ============================================================================

def callImage():
    """
    Charge et affiche toutes les images d'un dossier
    Appuyez sur 'q' pour quitter
    """
    # Chemin vers le dossier contenant les images
    path = r'C:\Dossier E\Henallux\M1\Systeme inteligent\Projet\flowers\Dataset_flowers\Training Data'

    # Trouver toutes les images .jpeg dans les sous-dossiers
    images = glob.glob(os.path.join(path, '**', '*.jpeg'), recursive=True)

    # Afficher chaque image avec un délai
    for img_path in images:
        img = cv2.imread(img_path)
        cv2.imshow('Image', img)
        key = cv2.waitKey(20)  # Attendre
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def get_image_folder() -> Path:
    """
    Récupère le chemin du dossier contenant les images de test
    """
    base_dir = Path(__file__).resolve().parent
    return base_dir.parent.parent / "GitHub" / "DeepLearning-Flowers" / "flowers" / "Dataset_flowers" / "Testing Data"


def choose_random_image(image_folder: Path) -> Path:
    """
    Choisit une image aléatoire dans le dossier
    """
    image_files = sorted(image_folder.rglob("*.jpeg")) + sorted(image_folder.rglob("*.jpg")) + sorted(image_folder.rglob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"Aucune image trouvée dans : {image_folder}")
    return random.choice(image_files)


def show_image(image_path: Path, delay: float = 0.5) -> None:
    """
    Affiche une image avec Matplotlib
    """
    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image.convert("RGB"))
    ax.axis('off')
    plt.show(block=False)
    plt.pause(delay)
    plt.close(fig)


# ============================================================================
# 8. EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    # Chemins vers les données
    chemin_train = r"C:\Dossier E\Henallux\M1\Systeme inteligent\Projet\flowers\Dataset_flowers\Training Data"
    chemin_test = r"C:\Dossier E\Henallux\M1\Systeme inteligent\Projet\flowers\Dataset_flowers\Testing Data"

    # Charge les données
    train, test, class_names = charger_donnees(chemin_train, chemin_test)

    # Affiche un aperçu
    afficher_apercu_images(train, class_names, nb_images=9)
