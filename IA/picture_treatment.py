# ============================================================================
# TRAITEMENT D'IMAGES - Preprocessing et augmentation pour classification
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras
from keras import layers
import tensorflow as tf
from PIL import Image, ImageEnhance
import os
import random
import glob
from pathlib import Path

# ============================================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================================

def charger_donnees(chemin_train, chemin_test, taille_image=256, batch_size_train=32, batch_size_test=32):
    train = keras.utils.image_dataset_from_directory(
        chemin_train,
        labels="inferred",
        batch_size=batch_size_train,
        image_size=(taille_image, taille_image),
        format="tf"
    )
    test = keras.utils.image_dataset_from_directory(
        chemin_test,
        labels="inferred",
        batch_size=batch_size_test,
        image_size=(taille_image, taille_image),
        format="tf",
        shuffle=False
    )
    class_names = train.class_names
    return train, test, class_names


# ============================================================================
# 2. VISUALISATION DES IMAGES
# ============================================================================

def afficher_apercu_images(dataset, class_names, nb_images=9):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(min(nb_images, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(class_names[int(labels[i])])
            plt.axis("off")
    plt.show()


# ============================================================================
# 3. VRAIE AUGMENTATION DE DONNÉES - Multiplie le dataset par N
# ============================================================================
#
# OPTIMISATIONS MÉMOIRE appliquées :
#   - num_parallel_calls=2  au lieu de AUTOTUNE (évite de surcharger la RAM)
#   - shuffle buffer_size=2000 au lieu de 10000 (moins de batches en mémoire)
#   - prefetch(2) au lieu de AUTOTUNE (limite le pré-chargement)
#
# Ces réglages évitent le "BFCAllocator ran out of memory" sur CPU sans GPU.
# Si tu as accès à un GPU avec assez de VRAM, tu peux remettre AUTOTUNE.
# ============================================================================

def _rotation_batch(imgs, angle_deg):
    """
    Rotation d'un batch float32 (B, H, W, C) — rank 4.
    Utilise tf.raw_ops directement sur le batch, aucune dépendance externe.
    """
    angle_rad = tf.cast(angle_deg * (3.14159265358979 / 180.0), tf.float32)
    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)

    cx = 128.0
    cy = 128.0

    transform = tf.reshape(tf.stack([
        cos_a,  -sin_a, cx - cx * cos_a + cy * sin_a,
        sin_a,   cos_a, cy - cx * sin_a - cy * cos_a,
        0.0,     0.0
    ]), [1, 8])

    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=imgs,                          # (B, 256, 256, 3) rank 4 ✓
        transforms=transform,
        output_shape=tf.constant([256, 256]),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0
    )
    return rotated


def _appliquer_variante_batch(images, labels, variante_id):
    """
    Applique une transformation DÉTERMINISTE sur un batch entier.
    images : (B, 256, 256, 3), float32, valeurs 0-255
    """
    imgs = tf.cast(images, tf.float32) / 255.0

    if variante_id == 0:
        pass

    elif variante_id == 1:
        imgs = tf.image.flip_left_right(imgs)

    elif variante_id == 2:
        imgs = tf.image.flip_up_down(imgs)

    elif variante_id == 3:
        imgs = _rotation_batch(imgs, angle_deg=15.0)

    elif variante_id == 4:
        imgs = _rotation_batch(imgs, angle_deg=-15.0)

    elif variante_id == 5:
        crop_size = int(256 * 0.80)  # 204
        offset = (256 - crop_size) // 2  # 26
        imgs = imgs[:, offset:offset + crop_size, offset:offset + crop_size, :]
        imgs = tf.image.resize(imgs, [256, 256])

    elif variante_id == 6:
        pad = 30
        imgs = tf.pad(imgs, [[0,0],[pad,pad],[pad,pad],[0,0]], mode='REFLECT')
        imgs = tf.image.resize(imgs, [256, 256])

    elif variante_id == 7:
        imgs = tf.image.adjust_brightness(imgs, delta=0.2)
        imgs = tf.clip_by_value(imgs, 0.0, 1.0)

    elif variante_id == 8:
        imgs = tf.image.adjust_brightness(imgs, delta=-0.2)
        imgs = tf.clip_by_value(imgs, 0.0, 1.0)

    elif variante_id == 9:
        imgs = tf.image.flip_left_right(imgs)
        imgs = _rotation_batch(imgs, angle_deg=10.0)

    return imgs, labels


def augmenter_dataset_reel(dataset, multiplicateur=1):
    """
    Crée un dataset augmenté en concaténant N variantes transformées.
    Le dataset final est multiplicateur × plus grand que l'original.

    Paramètres:
        dataset       : Dataset TF original (non normalisé, valeurs 0-255)
        multiplicateur: Nombre de copies (défaut 10 → ×10 images)

    Retour:
        dataset_augmente : Dataset concaténé, normalisé, mélangé
    """
    variantes = []

    for i in range(multiplicateur):
        vi = i % 10
        variante_i = dataset.map(
            lambda x, y, v=vi: _appliquer_variante_batch(x, y, v),
            num_parallel_calls=2   # ← Limité à 2 workers pour économiser la RAM
                                   #   (AUTOTUNE utilisait trop de mémoire : 61GB+)
        )
        variantes.append(variante_i)

    dataset_augmente = variantes[0]
    for v in variantes[1:]:
        dataset_augmente = dataset_augmente.concatenate(v)

    # Buffer réduit : 2000 au lieu de 10000
    # Chaque batch fait ~24MB (32 images × 256×256×3×4bytes)
    # 2000 batches = ~48GB → encore beaucoup, mais TF ne les charge pas tous d'un coup
    dataset_augmente = dataset_augmente.shuffle(
        buffer_size=500,
        reshuffle_each_iteration=True
    )

    # prefetch(2) : prépare seulement 2 batches à l'avance
    dataset_augmente = dataset_augmente.prefetch(2)

    return dataset_augmente


# ============================================================================
# 4. AUGMENTATION "À LA VOLÉE" (ancienne méthode - gardée pour compatibilité)
# ============================================================================

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
]

def augmentation_donnees(images):
    """(Ancienne méthode) Ne multiplie PAS le dataset."""
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


# ============================================================================
# 5. NORMALISATION
# ============================================================================

def normaliser_donnees(x_train, x_test):
    print('Before normalization : Min={}, max={}'.format(x_train.min(), x_train.max()))
    xmax = x_train.max()
    x_train = x_train / xmax
    x_test = x_test / xmax
    print('After normalization  : Min={}, max={}'.format(x_train.min(), x_train.max()))
    return x_train, x_test


# ============================================================================
# 6. TRAITEMENTS D'IMAGES AVANCÉS (inchangés)
# ============================================================================

def ajuster_contraste(image, facteur=1.5):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    return np.array(ImageEnhance.Contrast(image).enhance(facteur))


def ajuster_luminosite(image, facteur=1.2):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    return np.array(ImageEnhance.Brightness(image).enhance(facteur))


def appliquer_flou_gaussien(image, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def detecter_contours(image, method='canny'):
    if len(image.shape) == 3:
        image_grise = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_grise = image
    if method == 'canny':
        return cv2.Canny(image_grise, 100, 200)
    sobelx = cv2.Sobel(image_grise, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image_grise, cv2.CV_64F, 0, 1, ksize=5)
    return np.sqrt(sobelx**2 + sobely**2).astype('uint8')


def convertir_espace_couleur(image, espace='HSV'):
    conversions = {
        'HSV':   cv2.COLOR_RGB2HSV,
        'LAB':   cv2.COLOR_RGB2LAB,
        'GRAY':  cv2.COLOR_RGB2GRAY,
        'YCrCb': cv2.COLOR_RGB2YCrCb,
    }
    code = conversions.get(espace)
    return cv2.cvtColor(image, code) if code else image


def egaliser_histogramme(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def egaliser_histogramme_couleur(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_hsv[:, :, 2] = clahe.apply(image_hsv[:, :, 2])
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


def appliquer_filtre_median(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)


def redimensionner_image(image, largeur, hauteur):
    return cv2.resize(image, (largeur, hauteur))


# ============================================================================
# 7. PIPELINE COMPLET DE PREPROCESSING
# ============================================================================

def pipeline_preprocessing_complet(image, normaliser=True, augmenter=False):
    image = egaliser_histogramme_couleur(image)
    image = appliquer_filtre_median(image, kernel_size=5)
    image = ajuster_contraste(image, facteur=1.2)
    if normaliser:
        image = image.astype(np.float32) / 255.0
    if augmenter:
        image = augmentation_donnees(image)
    return image


# ============================================================================
# 8. FONCTIONS D'AFFICHAGE
# ============================================================================

def callImage():
    path = r'C:\Dossier E\Henallux\M1\Systeme inteligent\Projet\flowers\Dataset_flowers\Training Data'
    images = glob.glob(os.path.join(path, '**', '*.jpeg'), recursive=True)
    for img_path in images:
        img = cv2.imread(img_path)
        cv2.imshow('Image', img)
        if cv2.waitKey(20) == ord('q'):
            break
    cv2.destroyAllWindows()


def get_image_folder() -> Path:
    base_dir = Path(__file__).resolve().parent
    return base_dir.parent.parent / "GitHub" / "DeepLearning-Flowers" / "flowers" / "Dataset_flowers" / "Testing Data"


def choose_random_image(image_folder: Path) -> Path:
    image_files = (sorted(image_folder.rglob("*.jpeg")) +
                   sorted(image_folder.rglob("*.jpg")) +
                   sorted(image_folder.rglob("*.png")))
    if not image_files:
        raise FileNotFoundError(f"Aucune image trouvée dans : {image_folder}")
    return random.choice(image_files)


def show_image(image_path: Path, delay: float = 0.5) -> None:
    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image.convert("RGB"))
    ax.axis('off')
    plt.show(block=False)
    plt.pause(delay)
    plt.close(fig)


# ============================================================================
# 9. EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    chemin_train = r"C:\Dossier E\Henallux\M1\Systeme inteligent\Projet\flowers\Dataset_flowers\Training Data"
    chemin_test  = r"C:\Dossier E\Henallux\M1\Systeme inteligent\Projet\flowers\Dataset_flowers\Testing Data"
    train, test, class_names = charger_donnees(chemin_train, chemin_test)
    afficher_apercu_images(train, class_names, nb_images=9)