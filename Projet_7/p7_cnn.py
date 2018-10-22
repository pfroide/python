#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:19:42 2018

@author: toni
"""
import os
import random
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Lieu où se trouvent des images
IMG_DIR = '/home/toni/Bureau/p7/flow/'
IMG_DIR_TRAIN = '/home/toni/Bureau/p7/flow/train/'
IMG_DIR_TEST = '/home/toni/Bureau/p7/flow/test/'

# Définitions des limites d'execution
NB_RACES = 25
NB_EXEMPLES = 300
NB_CLUSTER = int(NB_RACES * (NB_EXEMPLES/5))

def calcul_resultats(res, test_y, classifieur):
    """
    Fonction qui va calculer les pourcentages de bons pronostics
    """

    print("\nResultats pour", classifieur)

    # Transformation en tableau exploitable
    res1 = res.values

    data_resultats = pd.DataFrame(index=res.index, columns=['bons',
                                                            'prono',
                                                            'total',
                                                            'pc_prono',
                                                            'pc_total'])

    # Affichage des résultats
    print("Resultat :", round(100*res1.diagonal().sum()/len(test_y), 2), "%")

    for i in range(0, len(res)):
        diagonale = res1.diagonal()[i]
        data_resultats.loc[res.index[i], 'bons'] = diagonale
        data_resultats.loc[res.index[i], 'prono'] = res.sum()[i]
        data_resultats.loc[res.index[i], 'total'] = res.sum('columns')[i]
        data_resultats.loc[res.index[i], 'pc_prono'] = round(100*diagonale/res.sum()[i], 2)
        data_resultats.loc[res.index[i], 'pc_total'] = round(100*diagonale/res.sum('columns')[i], 2)

    data_resultats = data_resultats.fillna(0)

    print(data_resultats)

def main():
    """
    TBD
    """

    from keras.models import Model

    etablir_liste_chiens()

    #liste_images.to_csv('/home/toni/Bureau/liste.csv')
    liste_images = pd.read_csv('/home/toni/Bureau/liste.csv')
    del liste_images['Unnamed: 0']

    # Séparation des datasets testing/training
    liste_train, liste_test = train_test_split(liste_images,
                                               test_size=0.2)

    liste_train = liste_train.reset_index(drop="True")
    liste_test = liste_test.reset_index(drop="True")

    batch_size = 32
    input_shape = (224, 224, 3)

    # Extract Features
    vgg = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

#    # Récupérer la sortie de ce réseau
#    x = vgg.output

#    # Ajouter la nouvelle couche fully-connected pour la classification à 10 classes
#    predictions = Dense(NB_RACES, activation='softmax')(x)

#    # Définir le nouveau modèle
#    new_model = Model(inputs=vgg.input, outputs=predictions)

#    for layer in new_model.layers:
#        layer.trainable = False

    # Sans data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # avec data augmentation
#    datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
#                                                           rescale=1./255,
#                                                           #width_shift_range=0.2,
#                                                           #height_shift_range=0.2,
#                                                           #shear_range=0.2,
#                                                           #zoom_range=0.2,
#                                                           #horizontal_flip=True,
#                                                           fill_mode='nearest')

    # Training
    train_generator = datagen.flow_from_dataframe(dataframe=liste_train,
                                                  directory=IMG_DIR,
                                                  x_col='liste',
                                                  y_col='labels',
                                                  has_ext=True,
                                                  target_size=(224, 224),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

    nImagesTrain = len(train_generator.classes)

    # Training
    train_features = np.zeros(shape=(nImagesTrain, 7, 7, 512))
    train_labels = np.zeros(shape=(nImagesTrain, NB_RACES))

    i = 0
    for inputs_batch, labels_batch in train_generator:
        features_batch = vgg.predict(inputs_batch)
        train_features[i * batch_size : (i + 1) * batch_size] = features_batch
        train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= nImagesTrain:
            break

    train_features = np.reshape(train_features, (nImagesTrain, 7 * 7 * 512))

    # Testing
    validation_generator = datagen.flow_from_dataframe(dataframe=liste_test,
                                                       directory=IMG_DIR,
                                                       x_col='liste',
                                                       y_col='labels',
                                                       has_ext=True,
                                                       target_size=(224, 224),
                                                       batch_size=batch_size,
                                                       class_mode='categorical')

    nImagesTest = len(validation_generator.classes)

    validation_features = np.zeros(shape=(nImagesTest, 7, 7, 512))
    validation_labels = np.zeros(shape=(nImagesTest, NB_RACES))

    i = 0
    for inputs_batch, labels_batch in validation_generator:
        features_batch = vgg.predict(inputs_batch)
        validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
        validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= nImagesTest:
            break

    validation_features = np.reshape(validation_features, (nImagesTest, 7 * 7 * 512))

    # Initialisation du CNN
    model = Sequential()

    # 1 - Convolution
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))

    # 2 - Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3 - Flattening
    model.add(Flatten())

    # 4 - Full Connection
    model.add(Dense(activation='relu', units=128))
    model.add(Dense(activation='relu', units=NB_RACES))

    # 5 - Compilation
#    model.compile(optimizer='adam',
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])

#    model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
#                  loss='categorical_crossentropy',
#                  metrics=['acc'])

    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # 6 - Fit
    # Train the model
    # Training a network in Keras is as simple as calling model.fit() function
    # as we have seen in our earlier tutorials.
    resultat = model.fit_generator(train_generator,
                                   epochs=20,
                                   verbose=1,
                                   validation_data=validation_generator)

    score = model.evaluate(validation_features, validation_labels, batch_size=32)

    # Check Performance
    # We would like to visualize which images were wrongly classified.
    fnames = validation_generator.filenames
    ground_truth = validation_generator.classes
    label2index = validation_generator.class_indices

    # Getting the mapping from class index to class label
    idx2label = dict((v, k) for k, v in label2index.items())

    predictions = model.predict_classes(validation_features)
    prob = model.predict(validation_features)

    # Calcul des résultats
    res = pd.crosstab(np.asarray(ground_truth),
                      predictions,
                      rownames=["Actual"],
                      colnames=["Predicted"])

    # Gestion d'une erreur
    if len(res.columns) != NB_RACES:
        res = gestion_erreur(res, predictions, liste_train['labels'], 'cnn')
    calcul_resultats(res, np.asarray(ground_truth), 'cnn')

    errors = np.where(predictions != ground_truth)[0]
    print("No of errors =", len(errors), "/", len(liste_test))

    # Let us see which images were predicted wrongly
    for i in range(len(errors)):
        pred_class = np.argmax(prob[errors[i]])
        pred_label = idx2label[pred_class]

        print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[errors[i]].split('/')[0],
            pred_label,
            prob[errors[i]][pred_class]))

        original = load_img('{}/{}'.format(IMG_DIR_TEST, fnames[errors[i]]))
        plt.imshow(original)
        plt.show()

def courbes():
    """
    TBD
    """

    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(resultat.history['loss'],'r',linewidth=3.0)
    plt.plot(resultat.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(resultat.history['acc'],'r',linewidth=3.0)
    plt.plot(resultat.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)

def gestion_erreur(res, test_y, labels, classifieur):
    """
    Gestion de l'erreur quand une catégorie de chien n'est pas prédite
    On rajoute la colonne vide manuellement
    """

    # Si ce n'est pas un kmeans, le traitement est différent (noms ou numéros)
    if classifieur == 'kmeans':
        for i in range(0, NB_RACES):
            if i not in res.columns:
                res[i] = 0

        for i in np.unique(labels):
            if i not in res.index:
                res.loc[i] = 0

    elif classifieur == 'cnn':
        for i in res.index:
            if i not in res.columns:
                res[i] = 0
    else:
        for i in np.unique(test_y):
            if i not in res.columns:
                res[i] = 0

        for i in res.columns:
            if i not in res.index:
                res.loc[i] = 0

    res = res.sort_index(axis=0, ascending=True)
    res = res.sort_index(axis=1, ascending=True)

    return res

def etablir_liste_chiens():
    """
    Création de la liste aléatoire des chiens pour les races selectionnés
    """

    dossier_source = '/home/toni/Bureau/p7/Images/'
    # Listes
    liste_dossier = []
    liste_images = []
    labels = []

    # Valeur initiale d'un compteur
    cpt_race = 0

    # Création de la liste aléatoire des races
    liste_chiens = os.listdir(dossier_source)
    #liste_chiens = [x for x in liste_chiens if "." not in x]

    for i in range(0, NB_RACES):
        nb_alea = random.randrange(0, len(liste_chiens))
        liste_dossier.append(liste_chiens[nb_alea])
        del liste_chiens[nb_alea]

    # Création de la liste aléatoire des chiens pour les races selectionnés
    for dirs in liste_dossier:
        # Valeur initiale d'un compteur
        cpt_exemple = 0
        if cpt_race < NB_RACES+1:
            cpt_race = cpt_race+1
            for filename in os.listdir(dossier_source + dirs):
                # On ne garde que NB_EXEMPLES exemplaires de chaque race
                if cpt_exemple < NB_EXEMPLES:
                    cpt_exemple = cpt_exemple+1

                    # Chemin complet de l'image
                    #liste_images.append(IMG_DIR + dirs + '/' + filename)
                    liste_images.append(filename)

                    # Rajout du label
                    labels.append(dirs[dirs.find('-')+1:].lower())

    liste_images = pd.DataFrame(liste_images, columns=['liste'])
    liste_images['labels'] = labels
    liste_images.to_csv('/home/toni/Bureau/liste.csv')

# =============================================================================
# def training():
#     """
#     TBD
#     """
#
#     dossier = '/home/toni/Bureau/'
#     image = dossier + '21.jpg'
#
#     # Création du modèle VGG-16 implementé par Keras
#     model = VGG16()
#
#     # VGG-16 reçoit des images de taille (224, 224, 3)
#     # la fonction  load_img  permet de charger l'image et de la redimensionner
#     # correctement
#     img = load_img(image, target_size=(224, 224))  # Charger l'image
#
#     # Keras traite les images comme des tableaux numpy
#     # img_to_array  permet de convertir l'image chargée en tableau numpy
#     img = img_to_array(img)  # Convertir en tableau numpy
#
#     # Le réseau doit recevoir en entrée une collection d'images,
#     # stockée dans un tableau de 4 dimensions, où les dimensions correspondent
#     # (dans l'ordre) à (nombre d'images, largeur, hauteur, profondeur).
#     # Pour l'instant, nous donnons qu'une image en entrée : numpy.reshape
#     # permet d'ajouter la première dimension (nombre d'images = 1) à notre image.
#     img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
#
#     # Enfin,  preprocess_input  permet d'appliquer les mêmes pré-traitements
#     # que ceux utilisés sur l'ensemble d'apprentissage lors du pré-entraînement.
#     img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16
#
#     # Prédire la classe de l'image (parmi les 1000 classes d'ImageNet)
#     y = model.predict(img)
#
#     # Afficher les 3 classes les plus probables
#     print('Top 3 :', decode_predictions(y, top=3)[0])
#
# def transfert_learning():
#     """
#     TBD
#     """
#
#     # Charger VGG-16 pré-entraîné sur ImageNet et sans les couches fully-connected
#     model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#
#     # Récupérer la sortie de ce réseau
#     x = model.output
#
#     # Ajouter la nouvelle couche fully-connected pour la classification à 10 classes
#     predictions = Dense(10, activation='softmax')(x)
#
#     # Définir le nouveau modèle
#     new_model = Model(inputs=model.input, outputs=predictions)
#
#     # Stratégie #1 : fine-tuning total
#     # Ici, on entraîne tout le réseau, donc il faut rendre toutes les
#     # couches "entraînables" :
#
#     for layer in model.layers:
#        layer.trainable = True
#
#     # Stratégie #2 : extraction de features
#     # On entraîne seulement le nouveau classifieur et on ne ré-entraîne pas
#     # les autres couches :
#
#     for layer in model.layers:
#        layer.trainable = False
#
#     # Stratégie #3 : fine-tuning partiel
#     # On entraîne le nouveau classifieur et les couches hautes :
#
#     # Ne pas entraîner les 5 premières couches (les plus basses)
#     for layer in model.layers[:5]:
#        layer.trainable = False
#
#     #  Entraînement du réseau
#     # Il ne reste plus qu'à compiler le nouveau modèle, puis à l'entraîner  :
#     # Compiler le modèle
#     new_model.compile(loss="categorical_crossentropy",
#                       optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
#                       metrics=["accuracy"])
#
#     # Entraîner sur les données d'entraînement (X_train, y_train)
#     model_info = new_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
#
# =============================================================================
