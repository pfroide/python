#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:19:42 2018

@author: toni
"""
import keras
import os
import tensorflow as tf

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

model = VGG16() # Création du modèle VGG-16 implementé par Keras

def training():
    """
    TBD
    """

    # VGG-16 reçoit des images de taille (224, 224, 3)
    # la fonction  load_img  permet de charger l'image et de la redimensionner
    # correctement
    img = load_img('/home/toni/Bureau/saul-5.jpg', target_size=(224, 224))  # Charger l'image

    # Keras traite les images comme des tableaux numpy
    # img_to_array  permet de convertir l'image chargée en tableau numpy
    img = img_to_array(img)  # Convertir en tableau numpy

    # Le réseau doit recevoir en entrée une collection d'images,
    # stockée dans un tableau de 4 dimensions, où les dimensions correspondent
    # (dans l'ordre) à (nombre d'images, largeur, hauteur, profondeur).
    # Pour l'instant, nous donnons qu'une image en entrée : numpy.reshape
    # permet d'ajouter la première dimension (nombre d'images = 1) à notre image.
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    # Enfin,  preprocess_input  permet d'appliquer les mêmes pré-traitements
    # que ceux utilisés sur l'ensemble d'apprentissage lors du pré-entraînement.
    img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16

    # Prédire la classe de l'image (parmi les 1000 classes d'ImageNet)
    y = model.predict(img)

    # Afficher les 3 classes les plus probables
    print('Top 3 :', decode_predictions(y, top=3)[0])


def transfert_learning():
    """
    TBD
    """

    # Stratégie #1 : fine-tuning total
    # Ici, on entraîne tout le réseau, donc il faut rendre toutes les
    # couches "entraînables" :

    for layer in model.layers:
       layer.trainable = True

    # Stratégie #2 : extraction de features
    # On entraîne seulement le nouveau classifieur et on ne ré-entraîne pas
    # les autres couches :

    for layer in model.layers:
       layer.trainable = False

    # Stratégie #3 : fine-tuning partiel
    # On entraîne le nouveau classifieur et les couches hautes :

    # Ne pas entraîner les 5 premières couches (les plus basses)
    for layer in model.layers[:5]:
       layer.trainable = False


    #  Entraînement du réseau
    # Il ne reste plus qu'à compiler le nouveau modèle, puis à l'entraîner  :
    # Compiler le modèle
    new_model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

    # Entraîner sur les données d'entraînement (X_train, y_train)
    model_info = new_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)