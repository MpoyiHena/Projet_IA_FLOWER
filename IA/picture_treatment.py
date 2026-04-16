import random
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def get_image_folder() -> Path:
    base_dir = Path(__file__).resolve().parent
    return base_dir.parent.parent / "GitHub" / "DeepLearning-Flowers" / "flowers" / "Dataset_flowers" / "Testing Data"

def choose_random_image(image_folder: Path) -> Path:
    image_files = sorted(image_folder.rglob("*.jpeg")) + sorted(image_folder.rglob("*.jpg")) + sorted(image_folder.rglob("*.png"))
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

if __name__ == "__main__":
    folder = get_image_folder()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {folder}")

    for i in range(10):
        image_path = choose_random_image(folder)
        print(f"Affichage de l'image {i+1} : {image_path.name}")
        show_image(image_path, delay=0.5)
        if i == 0:
            time.sleep(0.5)


