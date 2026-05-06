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
from keras.callbacks import EarlyStopping

from picture_treatment import (
    charger_donnees,
    afficher_apercu_images,
    augmentation_donnees,
    normaliser_donnees,
    pipeline_preprocessing_complet
)

# ============================================================================
# ETAPE 1 : CHARGEMENT ET VISUALISATION DES DONNEES
# ============================================================================

chemin_train = r"/srv/groups/group4/data/Training Data"
chemin_test  = r"/srv/groups/group4/data/Testing Data"

train, test, class_names = charger_donnees(
    chemin_train,
    chemin_test,
    batch_size_train=32,
    batch_size_test=32
)

afficher_apercu_images(train, class_names, nb_images=9)

# ============================================================================
# ETAPE 2 : DATA AUGMENTATION ET NORMALISATION
# ============================================================================

train = train.map(lambda x, y: (x / 255.0, y))
test  = test.map(lambda x, y: (x / 255.0, y))

train = train.map(lambda x, y: (augmentation_donnees(x), y))

# ============================================================================
# ETAPE 3 : CRÉATION DU MODELE CNN
# ============================================================================

model = keras.models.Sequential()

# ---- BLOC CONVOLUTIF 1 ----
model.add(keras.Input(shape=(256, 256, 3)))
model.add(keras.layers.Conv2D(16, (5,5), activation='relu', padding="same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.2))

# ---- BLOC CONVOLUTIF 2 ----
model.add(keras.layers.Conv2D(32, (5,5), activation='relu', padding="same"))  # 32 filtres (pas 16 !)
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.2))

# ---- BLOC CONVOLUTIF 3 ----
model.add(keras.layers.Conv2D(64, (5,5), activation='relu', padding="same"))
model.add(keras.layers.BatchNormalization())  # manquait dans ton fichier
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.2))

# ---- COUCHES DE CLASSIFICATION ----
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))

# Couche de sortie : 11 classes
model.add(keras.layers.Dense(11, activation='softmax'))

model.summary()

# ---- COMPILATION ----
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================================
# ETAPE 4 : ENTRAINEMENT
# ============================================================================

batch_size = 32
epochs     = 30

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train,
    batch_size      = batch_size,
    epochs          = epochs,
    verbose         = 1,
    validation_data = test,
    callbacks       = [early_stop]  # décommenté !
)

# ============================================================================
# ETAPE 5 : EVALUATION
# ============================================================================

score = model.evaluate(test, verbose=0)
print(f'Test loss     : {score[0]:4.4f}')
print(f'Test accuracy : {score[1]:4.4f}')
print('Entraînement et évaluation terminés.')

# ============================================================================
# ETAPE 6 : COURBES D'APPRENTISSAGE
# ============================================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Val accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('courbes_apprentissage.png')
plt.show()