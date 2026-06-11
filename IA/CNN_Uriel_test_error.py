# ============================================================================
# CLASSIFICATION D'IMAGES DE FLEURS AVEC UN RESEAU DE NEURONES CONVOLUTIF (CNN)
# ============================================================================
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import cv2
import keras
from keras import layers
import os
import sys
from datetime import datetime
import json
from keras.callbacks import EarlyStopping, ModelCheckpoint
 
# Ensure the local IA/powerwork.py module is loaded first.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import powerwork as pwk
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
import seaborn as sns
# import scikitplot as skplt
 
from picture_treatment import (
    charger_donnees,
    afficher_apercu_images,
    augmentation_donnees,
    normaliser_donnees,
    pipeline_preprocessing_complet,
    augmenter_dataset_reel          # ← NOUVELLE FONCTION importée
)
 
print(tf.config.list_physical_devices('GPU'))

# ============================================================================
# ETAPE 1 : CHARGEMENT ET VISUALISATION DES DONNEES
# ============================================================================
 
chemin_train = r"/srv/groups/group4/data/Training Data"
chemin_test  = r"/srv/groups/group4/data/Testing Data"
 
train, test, class_names = charger_donnees(
    chemin_train,
    chemin_test,
    batch_size_train=50,   # Augmentation du batch size pour accélérer l'entraînement avec plus de données
    batch_size_test=50
)
 
# ============================================================================
# ETAPE 2 : VRAIE AUGMENTATION DE DONNÉES (×10)
# ============================================================================
#
# AVANT (ancienne méthode) :
#   train = train.map(lambda x, y: (x / 255.0, y))
#   train = train.map(lambda x, y: (augmentation_donnees(x), y))
#   → Chaque image est REMPLACÉE par une version transformée aléatoirement
#   → Le dataset garde 15 746 images → toujours ~493 iterations/epoch
#
# MAINTENANT (nouvelle méthode) :
#   augmenter_dataset_reel() crée 10 COPIES du dataset avec des
#   transformations DIFFÉRENTES et les CONCATÈNE.
#   → 15 746 × 10 = 157 460 images → ~4920 iterations/epoch  ✓
#
# Les 10 transformations appliquées (une par copie) :
#   0 - Original              5 - Zoom avant
#   1 - Flip horizontal       6 - Zoom arrière
#   2 - Flip vertical         7 - Luminosité +20%
#   3 - Rotation +15°         8 - Luminosité -20%
#   4 - Rotation -15°         9 - Flip H + Rotation 10°
#
# La normalisation (÷255) est faite à l'intérieur de augmenter_dataset_reel.
# ============================================================================
 
MULTIPLICATEUR = 10  # Changer ici pour ×5, ×15, etc.
 
print(f"\n[INFO] Augmentation du dataset d'entraînement ×{MULTIPLICATEUR}...")
train_augmente = augmenter_dataset_reel(train, multiplicateur=MULTIPLICATEUR)
print(f"[INFO] Dataset prêt. Iterations attendues par epoch : ~{15746 * MULTIPLICATEUR // 32}")
 
# Normalisation du dataset de test uniquement (pas d'augmentation sur le test)
test = test.map(lambda x, y: (x / 255.0, y))
test = test.prefetch(tf.data.AUTOTUNE)
 
 
# ============================================================================
# ETAPE 3 : CRÉATION DU MODELE CNN
# ============================================================================
 
model = keras.models.Sequential()
 
# ---- BLOC CONVOLUTIF 1 ----
 
model.add(keras.Input(shape=(256, 256, 3)))
model.add(keras.layers.Conv2D(16, (5, 5), activation='relu', padding="same"))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.4))
 
# ---- BLOC CONVOLUTIF 2 ----
model.add(keras.layers.Conv2D(32, (5, 5), activation='relu', padding="same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.4))
 
# ---- BLOC CONVOLUTIF 3 ----
model.add(keras.layers.Conv2D(64, (5, 5), activation='relu', padding="same"))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.4))
 
# ---- COUCHES DE CLASSIFICATION ----
#model.add(keras.layers.Flatten())
model.add(keras.layers.GlobalAveragePooling2D())  # ← Réduit drastiquement les params
model.add(keras.layers.Dense(132, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(11, activation='softmax'))
 
model.summary()
 
model.compile(
    optimizer='adamw',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
 
# ============================================================================
# ETAPE 4 : ENTRAINEMENT DU MODELE CNN
# ============================================================================
 
epochs = 50  # Augmentation du nombre d'epochs pour compenser la plus grande quantité de données

# Création d'un dossier horodaté pour versionner cette exécution
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.getcwd(), "runs", timestamp)
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Sauvegarde des sorties dans: {output_dir}")

# Checkpoint vers le dossier de sortie horodaté
checkpoint = ModelCheckpoint(
    os.path.join(output_dir, 'best_model_flowers.h5'),
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
 
# On passe train_augmente (×10) au lieu de train
history = model.fit(
    train_augmente,             # ← dataset augmenté ×10
    epochs=epochs,
    verbose=1,
    validation_data=test,
    callbacks=[checkpoint]     # Enregistre le meilleur modèle pendant les 50 epochs
)

# Sauvegarder l'historique d'entraînement
with open(os.path.join(output_dir, 'history.json'), 'w') as f:
    json.dump(history.history, f)
 
 
# ============================================================================
# ETAPE 5 : EVALUATION DU MODELE CNN
# ============================================================================
 
score = model.evaluate(test, verbose=0)
print(f'Test loss     : {score[0]:4.4f}')
print(f'Test accuracy : {score[1]:4.4f}')
print('Entraînement et évaluation terminés.')
 
model.save(os.path.join(output_dir, 'my_model.keras'))
 
 
# ---------- Extraire les données de test en tableau pour les visualisations ----------
# Note : test est un tf.data.Dataset en batch ; on le convertit en tableaux numpy.
x_test = np.concatenate([x.numpy() for x, y in test], axis=0)
y_test = np.concatenate([y.numpy() for x, y in test], axis=0)

# Convertir les labels de test au format 1D si nécessaire
if y_test.ndim == 2 and y_test.shape[1] == 1:
    y_test = y_test.squeeze(axis=-1)
elif y_test.ndim > 1:
    y_test = np.argmax(y_test, axis=-1)

# ---------- Plots d'entraînement : accuracy et loss ----------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy pendant l\'entraînement')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
 
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss pendant l\'entraînement')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_accuracy_loss.png'), dpi=200, bbox_inches='tight')
plt.show()
 
# ---------- Prédiction sur le jeu de test ----------
y_pred_proba = model.predict(x_test, verbose=0)
# Label prédit par argmax (sans tenir compte de la confiance)
y_pred = np.argmax(y_pred_proba, axis=-1)

# Affichage rapide des premières prédictions (selon l'ancienne logique)
pwk.plot_images(x_test, y_test, range(0, min(200, len(x_test))), columns=12, x_size=1, y_size=1, y_pred=y_pred, save_as=os.path.join(output_dir, '04-predictions'))

# --- Détection des prédictions peu confiantes ("Unknown") ---
# Seuil de confiance sous lequel on considère que le modèle "ne sait pas"
CONF_THRESHOLD = 0.50  # ajuster si besoin (par ex. 0.6 pour être plus strict)

# Probabilité maximale et prédiction finale tenant compte du seuil
max_probs = np.max(y_pred_proba, axis=1)
pred_argmax = np.argmax(y_pred_proba, axis=1)

# Définir un indice dédié pour "Unknown" qui n'entre pas en conflit
if isinstance(class_names, (list, tuple)):
    n_known_classes = len(class_names)
else:
    # si class_names inattendu, estimer depuis y_test
    n_known_classes = int(np.max(y_test)) + 1
unknown_label = n_known_classes

# y_pred_with_unknown contient unknown_label quand max_probs < CONF_THRESHOLD
y_pred_with_unknown = np.where(max_probs < CONF_THRESHOLD, unknown_label, pred_argmax)

# Créer dossier spécifique pour les prédictions inconnues
unknown_dir = os.path.join(output_dir, 'errors', 'unknown_predictions')
os.makedirs(unknown_dir, exist_ok=True)

# Sauvegarde rapide des images jugées "Unknown" (limite à éviter d'enregistrer trop de fichiers)
unknown_indices = np.where(max_probs < CONF_THRESHOLD)[0]
for idx in unknown_indices:
    try:
        img = x_test[idx]
        prob = float(max_probs[idx])
        pred_label = int(pred_argmax[idx])
        pred_name = class_names[pred_label] if isinstance(class_names, (list, tuple)) else str(pred_label)
        safe_pred = str(pred_name).replace(' ', '_')
        filename = f"{idx:05d}_unknown_prob-{prob:.2f}_pred-{safe_pred}.png"
        filepath = os.path.join(unknown_dir, filename)
        plt.imsave(filepath, np.clip(img, 0, 1))
    except Exception as e:
        print(f"Erreur en sauvegardant unknown index {idx}: {e}")

# Erreurs : indices où la prédiction (incluant 'Unknown') diffère de la vérité
erreurs_index = np.where(y_pred_with_unknown != y_test)[0]
errors = erreurs_index[:min(24, len(erreurs_index))]

# --- Préparation et affichage des erreurs ---
# incorrect_indices : indices dans x_test/y_test où la prédiction diffère de la vérité terrain
incorrect_indices = erreurs_index
print(f"Nombre d'erreurs trouvées : {len(incorrect_indices)}")

# Dossier pour stocker les images mal classées (ex: runs/20230601_120000/errors)
errors_dir = os.path.join(output_dir, 'errors')
os.makedirs(errors_dir, exist_ok=True)

# Parcours des indices incorrects et sauvegarde des images
for idx in incorrect_indices:
    try:
        # Récupère l'image correspondante depuis le tableau numpy x_test
        img = x_test[idx]

        # Détermination des labels vrais et prédits (format entier)
        # y_test peut être soit un entier, soit un one-hot / vecteur — on normalise ici
        true_label = int(y_test[idx]) if hasattr(y_test[idx], '__int__') else int(np.argmax(y_test[idx]))
        # Utiliser la prédiction prenant en compte le seuil de confiance
        pred_label = int(y_pred_with_unknown[idx])

        # Récupère le nom de classe lisible si disponible
        true_name = class_names[true_label] if isinstance(class_names, (list, tuple)) else str(true_label)
        if pred_label == unknown_label:
            pred_name = 'Unknown'
        else:
            pred_name = class_names[pred_label] if isinstance(class_names, (list, tuple)) else str(pred_label)

        # Prépare un nom de fichier sûr (pas d'espaces)
        safe_true = str(true_name).replace(' ', '_')
        safe_pred = str(pred_name).replace(' ', '_')
        filename = f"{idx:05d}_true-{safe_true}_pred-{safe_pred}.png"
        filepath = os.path.join(errors_dir, filename)

        # Sauvegarde l'image. Si les pixels sont normalisés [0,1], on clip pour être sûr.
        plt.imsave(filepath, np.clip(img, 0, 1))

    except Exception as e:
        # Ne pas planter l'évaluation si une image pose problème — on log l'erreur
        print(f"Erreur en sauvegardant l'index {idx}: {e}")
# Construire la matrice de confusion en incluant le label 'Unknown'
if isinstance(class_names, (list, tuple)):
    class_names_with_unknown = list(class_names) + ['Unknown']
else:
    class_names_with_unknown = [str(i) for i in range(n_known_classes)] + ['Unknown']
labels_for_cm = list(range(n_known_classes)) + [unknown_label]
cm = confusion_matrix(y_test, y_pred_with_unknown, labels=labels_for_cm)
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names_with_unknown,
    yticklabels=class_names_with_unknown,
    linewidths=0.5,
    linecolor='gray'
)
plt.xlabel('Classe prédite')
plt.ylabel('Classe réelle')
plt.title('Matrice de confusion par fleur')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=200, bbox_inches='tight')
plt.show()
