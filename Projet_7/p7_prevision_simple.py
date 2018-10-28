#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 22:39:52 2018

@author: toni
"""

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19

def main(fichier):
    """
    Fichier principale
    """

    # On crée un modèle déjà pré entrainé
    model = VGG19(weights="imagenet", include_top=True)

    # Visualisation de toutes les couches du modèle
    model.summary()

    # On va chercher l'imagenversion d
    image = load_img(fichier, target_size=(224, 224))

    # Convertion de l'image
    image = img_to_array(image)

    # Reshape en 4 dimensions
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Preprocessing
    image = preprocess_input(image)

    # Prediction (des probabilités)
    proba = model.predict(image)
    label = decode_predictions(proba)

    # Label avec la plus grande probabilité
    label = label[0][0]

    # Affichage
    print('Race :', label[1], 'avec une probabilité de', round((label[2]*100), 2), '%')
