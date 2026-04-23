import cv2
import numpy as np
from pathlib import Path
import time

# Importer les fonctions de picture_treatment.py
from picture_treatment import get_image_folder

def load_and_calibrate_images(image_folder: Path, target_width: int = 640, target_height: int = 480):
    """
    Charge toutes les images d'un dossier et les redimensionne à la même taille.
    
    Args:
        image_folder: Chemin vers le dossier contenant les images
        target_width: Largeur cible (défaut 640)
        target_height: Hauteur cible (défaut 480)
    
    Returns:
        Liste des images redimensionnées et une liste des noms de fichiers
    """
    image_files = sorted(image_folder.rglob("*.jpeg")) + \
                  sorted(image_folder.rglob("*.jpg")) + \
                  sorted(image_folder.rglob("*.png"))
    
    if not image_files:
        print(f"Aucune image trouvée dans : {image_folder}")
        return [], []
    
    calibrated_images = []
    filenames = []
    
    for image_path in image_files:
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Impossible de charger : {image_path}")
                continue
            
            # Redimensionner l'image au même cadrage
            # Option 1 : Redimensionner directement (déforme l'image)
            resized = cv2.resize(img, (target_width, target_height))
            
            # Option 2 (Commenté) : Garder les proportions avec fond blanc
            # height, width = img.shape[:2]
            # scale = min(target_width / width, target_height / height)
            # new_width = int(width * scale)
            # new_height = int(height * scale)
            # resized = cv2.resize(img, (new_width, new_height))
            # # Créer une image blanche de la taille cible
            # canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
            # # Centrer l'image redimensionnée
            # x_offset = (target_width - new_width) // 2
            # y_offset = (target_height - new_height) // 2
            # canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            # resized = canvas
            
            calibrated_images.append(resized)
            filenames.append(image_path.name)
            print(f"✓ Chargée : {image_path.name}")
        
        except Exception as e:
            print(f"Erreur lors du chargement de {image_path} : {e}")
    
    return calibrated_images, filenames

def display_calibrated_images(calibrated_images: list, filenames: list, delay: float = 2.0):
    """
    Affiche les images calibrées avec un délai configurable.
    
    Args:
        calibrated_images: Liste des images redimensionnées
        filenames: Liste des noms de fichiers
        delay: Délai en secondes entre les images (défaut 2.0)
    """
    if not calibrated_images:
        print("Aucune image à afficher")
        return
    
    print(f"\nAffichage de {len(calibrated_images)} images...")
    print("Appuyez sur 'q' ou 'ESC' pour arrêter, 'n' pour suivant, 'p' pour précédent")
    
    idx = 0
    while idx < len(calibrated_images):
        img = calibrated_images[idx]
        filename = filenames[idx]
        
        # Ajouter le nom de fichier sur l'image
        cv2.putText(img, f"{filename} ({idx + 1}/{len(calibrated_images)})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Images calibrées', img)
        
        key = cv2.waitKey(int(delay * 1000))
        
        if key == ord('q') or key == 27:  # 'q' ou ESC
            break
        elif key == ord('n'):  # 'n' pour suivant
            idx += 1
        elif key == ord('p'):  # 'p' pour précédent
            idx = max(0, idx - 1)
        else:
            idx += 1
    
    cv2.destroyAllWindows()

def display_calibrated_images_grid(calibrated_images: list, filenames: list, grid_size: int = 2):
    """
    Affiche les images calibrées en grille.
    
    Args:
        calibrated_images: Liste des images redimensionnées
        filenames: Liste des noms de fichiers
        grid_size: Nombre de colonnes (défaut 2)
    """
    if not calibrated_images:
        print("Aucune image à afficher")
        return
    
    for i in range(0, len(calibrated_images), grid_size):
        batch = calibrated_images[i:i+grid_size]
        batch_names = filenames[i:i+grid_size]
        
        # Combiner les images horizontalement
        combined = cv2.hconcat(batch)
        
        # Afficher le batch
        cv2.imshow(f'Grille d\'images ({i//grid_size + 1}/{(len(calibrated_images)-1)//grid_size + 1})', combined)
        
        key = cv2.waitKey(3000)
        if key == ord('q') or key == 27:
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Obtenir le dossier des images
    folder = get_image_folder()
    
    if not folder.exists() or not folder.is_dir():
        print(f"Dossier introuvable : {folder}")
    else:
        # Charger et calibrer les images
        images, names = load_and_calibrate_images(folder, target_width=640, target_height=480)
        
        if images:
            print(f"\n{len(images)} images chargées et calibrées avec succès !")
            # Afficher les images une par une
            display_calibrated_images(images, names, delay=2.0)
            # Optionnel : Afficher en grille
            # display_calibrated_images_grid(images, names, grid_size=2)
