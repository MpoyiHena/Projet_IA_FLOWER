# ============================================================================
# CLASSIFICATION D'IMAGES DE FLEURS AVEC UN RESEAU DE NEURONES CONVOLUTIF (CNN)
# ============================================================================

# Importation des bibliothèques nécessaires
import numpy as np  # Pour les opérations mathématiques et tableaux
import matplotlib.pyplot as plt  # Pour la visualisation des images
import pandas as pd  # Pour la manipulation de données (optionnel ici)
import tensorflow as tf  # Framework de deep learning
import cv2  # Traitement d'images
import keras  # API Keras pour construire les modèles
from keras import layers  # Couches de réseau de neurones
import os  # Pour les opérations sur les fichiers

# Import du module de traitement d'images
from picture_treatment import (
    charger_donnees,
    afficher_apercu_images,
    augmentation_donnees,
    normaliser_donnees,
    pipeline_preprocessing_complet
)

# Référence : https://keras.io/api/data_loading/image/

# ============================================================================
# ETAPE 1 : CHARGEMENT ET VISUALISATION DES DONNEES
# ============================================================================

# Chemins vers les données
chemin_train = r"C:\Dossier E\Henallux\M1\Systeme inteligent\Projet\flowers\Dataset_flowers\Training Data"
chemin_test = r"C:\Dossier E\Henallux\M1\Systeme inteligent\Projet\flowers\Dataset_flowers\Testing Data"

# Charge les données (voir picture_treatment.py pour les détails)
train, test, class_names = charger_donnees(chemin_train, chemin_test)

# Affiche un aperçu des images pour visualiser les données
afficher_apercu_images(train, class_names, nb_images=9)


# ============================================================================
# ETAPE 2 : DATA AUGMENTATION ET NORMALISATION
# ============================================================================
# Ces fonctions sont importées depuis picture_treatment.py :
# - augmentation_donnees() : applique flipping, rotation, zoom aléatoires
# - normaliser_donnees() : divise les valeurs par le max pour avoir 0-1
# - pipeline_preprocessing_complet() : applique tous les traitements ensemble

# Applique la normalisation : divise les valeurs de pixels par 255 pour avoir 0-1
train = train.map(lambda x, y: (x / 255.0, y))
test = test.map(lambda x, y: (x / 255.0, y))

# Applique l'augmentation de données sur l'ensemble d'entraînement uniquement
train = train.map(lambda x, y: (augmentation_donnees(x), y))


# ============================================================================
# ETAPE 3 : CRÉATION DU MODELE CNN
# ============================================================================
# Architecture : 3 couches convolutives + 2 couches denses + Dropout

# Crée un modèle séquentiel (couches empilées linéairement)
model = keras.models.Sequential()

# ---- BLOC CONVOLUTIF 1 ----
# 16 filtres, kernel 5x5, activation ReLU, padding "same" conserve la taille
model.add( keras.layers.Conv2D(
    16, (5,5),
    activation='relu',
    padding="same",
    input_shape = (256,256,3)
) )  # 16 filtres de convolution 2D, kernel 5x5
# MaxPooling réduit la taille de l'image (2x2) en prenant le max
model.add( keras.layers.MaxPooling2D((2,2)))
# Dropout optionnel : désactive aléatoirement 20% des neurones (régularisation)
# model.add( keras.layers.Dropout(0.2))

# ---- BLOC CONVOLUTIF 2 ----
# 32 filtres, kernel 5x5, activation ReLU
model.add( keras.layers.Conv2D(32, (5,5), activation='relu', padding="same") )
# MaxPooling réduit à nouveau la taille par 4
model.add( keras.layers.MaxPooling2D((2,2)))
# model.add( keras.layers.Dropout(0.2))

# ---- BLOC CONVOLUTIF 3 ----
# 64 filtres, kernel 5x5, activation ReLU
model.add( keras.layers.Conv2D(64, (5,5), activation='relu', padding="same") )
# MaxPooling réduit la taille
model.add( keras.layers.MaxPooling2D((2,2)))
# model.add( keras.layers.Dropout(0.2))

# ---- COUCHES DE CLASSIFICATION ----
# Aplatit le tenseur 3D en vecteur 1D pour les couches denses
model.add( keras.layers.Flatten())

# Couche dense cachée avec 128 neurones et activation ReLU
model.add( keras.layers.Dense(128, activation='relu'))
# Dropout 30% pour la régularisation
model.add( keras.layers.Dropout(0.3))

# Couche dense cachée avec 64 neurones et activation ReLU
model.add( keras.layers.Dense(64, activation='relu'))
# Dropout 50% pour éviter l'overfitting
model.add( keras.layers.Dropout(0.5))

# Couche de sortie : 10 neurones (nombre de classes de fleurs)
# Softmax convertit en probabilités (somme = 1)
model.add( keras.layers.Dense(10, activation='softmax'))

# Affiche un résumé de l'architecture du modèle
model.summary()

# ---- COMPILATION DU MODELE ----
model.compile(
    optimizer='adam',  # Adam : optimiseur pour descendre le gradient
    loss='sparse_categorical_crossentropy',  # Fonction de perte pour classification multi-classe
    metrics=['accuracy']  # Métrique à suivre
)

# ============================================================================
# ETAPE 4 : ENTRAINEMENT DU MODELE CNN
# ============================================================================
# Hyperparamètres d'entraînement
batch_size  = 32  # Nombre d'images traitées avant une mise à jour des poids
epochs      =  10  # Nombre de passages sur l'ensemble d'entraînement
history = model.fit(
    train,
    batch_size      = batch_size,
    epochs          = epochs,
    verbose         = 1,  # Affiche la progression
    validation_data = test  # Valide sur les données de test à chaque epoch
)
# ============================================================================
# ETAPE 5 : EVALUATION DU MODELE CNN
# ============================================================================

# Évalue le modèle sur l'ensemble de test
# Retourne [loss, accuracy]
score = model.evaluate(test, verbose=0)

# Affiche les résultats de l'évaluation
print(f'Test loss     : {score[0]:4.4f}')  # Erreur moyenne sur les données de test
print(f'Test accuracy : {score[1]:4.4f}')  # Précision (pourcentage correct)