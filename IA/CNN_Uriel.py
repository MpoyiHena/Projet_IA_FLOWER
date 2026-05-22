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
from keras.callbacks import EarlyStopping
 
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
model.add(keras.layers.Conv2D(8, (5, 5), activation='relu', padding="same"))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.2))
 
# ---- BLOC CONVOLUTIF 2 ----
model.add(keras.layers.Conv2D(16, (5, 5), activation='relu', padding="same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.2))
 
# ---- BLOC CONVOLUTIF 3 ----
model.add(keras.layers.Conv2D(32, (5, 5), activation='relu', padding="same"))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.2))
 
# ---- COUCHES DE CLASSIFICATION ----
model.add(keras.layers.Flatten())
#model.add(keras.layers.GlobalAveragePooling2D())  # ← Réduit drastiquement les params
model.add(keras.layers.Dense(64, activation='relu'))
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
 
epochs = 50
 
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
 
# On passe train_augmente (×5) au lieu de train
history = model.fit(
    train_augmente,             # ← dataset augmenté ×10
    epochs=epochs,
    verbose=1,
    validation_data=test,
    #callbacks=[early_stop]      # EarlyStopping activé (avec ×10 data)
)
 
 
# ============================================================================
# ETAPE 5 : EVALUATION DU MODELE CNN
# ============================================================================
 
score = model.evaluate(test, verbose=0)
print(f'Test loss     : {score[0]:4.4f}')
print(f'Test accuracy : {score[1]:4.4f}')
print('Entraînement et évaluation terminés.')

model.save('model_flowers.h5')
 
 
 
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
plt.savefig('training_accuracy_loss.png', dpi=200, bbox_inches='tight')
plt.show()
 
# ---------- Prédiction sur le jeu de test ----------
y_pred_proba = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=-1)
 
pwk.plot_images(x_test, y_test, range(0, min(200, len(x_test))), columns=12, x_size=1, y_size=1, y_pred=y_pred, save_as='04-predictions')
 
errors = [i for i in range(len(x_test)) if y_pred[i] != y_test[i]]
errors = errors[:min(24, len(errors))]
pwk.plot_images(x_test, y_test, errors, columns=6, x_size=2, y_size=2, y_pred=y_pred, save_as='05-some-errors')
 
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
print('Global accuracy :', accuracy_score(y_test, y_pred))
 
# ---------- Heatmap de la matrice de confusion par fleur ----------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor='gray'
)
plt.xlabel('Classe prédite')
plt.ylabel('Classe réelle')
plt.title('Matrice de confusion par fleur')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
