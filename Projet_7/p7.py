#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 21:38:27 2018

@author: toni
"""

import os
import random
import numpy as np
from scipy import misc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

## Import to a python dictionary
#mat = scipy.io.loadmat('/home/toni/Bureau/p7/test_data.mat')
#
## Look at the dictionary items
#mat.items()
#
## Print the data
#m = mat["test_data"]

# =============================================================================
# Une approche classique : il s’agit de pre-processer des images avec des
# techniques spécifiques (e.g. whitening, equalisation,
# filtre linéaire/laplacien/gaussien, éventuellement modifier la taille des images),
# puis d’extraire des features (e.g. texture, corners, edges et SIFT detector).
# Il faut ensuite réduire les dimensions, soit par des approches classiques
# (e.g. PCA, k-means) soit avec une approche par histogrammes et dictionary learning
# (bag-of-words appliqué aux images), puis appliquer des algorithmes de
# classification standards.
#
# Lors de l’analyse exploratoire, vous regarderez si les features extraites
# et utilisées en classification sont prometteuses en utilisant des
# méthodes de réduction de dimension pour visualiser le dataset en 2D.
# Cela vous permettra d’affiner votre intuition sur les différents traitements
# possibles, sans que cela ne se substitue à des mesures de performances
# rigoureuses.
# =============================================================================

#setup a standard image size; this will distort some images but will get everything into the same shape
STANDARD_SIZE = (300, 167)
#STANDARD_SIZE = (500, 375)

def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """

    img = Image.open(filename)

    #if verbose==True:
    #    print("changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE)))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    #img = map(list, img)
    img = np.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it
    into an array of shape (1, m * n)
    """

    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

def main():
    """
    TBD
    """

    # Définitions des limites d'execution
    NB_RACES = 8
    NB_EXEMPLES = 100

    img_dir = '/home/toni/Bureau/p7/Images/' # n02090379-redbone/'
    #images = [img_dir+ f for f in os.listdir(img_dir)]
    #labels = ["check" if "check" in f.split('/')[-1] else "redbone" for f in images]

#    # Création du dataframe vide
#    spec_images = pd.DataFrame()

    # Partie pour récupérer les tailles des images.
    # Pas forcément utile
#    for path, dirs, files in os.walk(img_dir):
#        for filename in files:
#            image = misc.imread(path + '/' + filename)
#            titre = path + '/' + filename
#            spec_images.loc[titre, 0] = image.shape[0]
#            spec_images.loc[titre, 1] = image.shape[1]
#            spec_images.loc[titre, 2] = image.shape[2]
#            spec_images.loc[titre, 3] = str(image.shape[0]) + '-' + \
#                                        str(image.shape[1]) + '-' + \
#                                        str(image.shape[2])
#
#    df = spec_images.groupby(3)[0].count()

    # Valeur initiale d'un compteur
    cpt_race = 0

    # Création des listes vides
    data = []
    labels = []

    liste_dossier = []
    p = os.listdir(img_dir)

    # Création de la liste aléatoire des races
    for i in range(0, NB_RACES):
        nb_alea = random.randrange(0, len(p))
        liste_dossier.append(p[nb_alea])
        del p[nb_alea]

    for dirs in liste_dossier:
        # Valeur initiale d'un compteur
        cpt_exemple = 0
        if cpt_race < NB_RACES+1:
            cpt_race = cpt_race+1
            for filename in os.listdir(img_dir + dirs):
                # On ne garde que 10 exemplaires de chaque race
                if cpt_exemple < NB_EXEMPLES:
                    cpt_exemple = cpt_exemple+1

                    # Affichage pour voir si ça tourne toujours
                    if cpt_exemple%25 == 0:
                        print(cpt_exemple, dirs + '/' + filename)

                    # Récupération de la matrice tranformée
                    img = img_to_matrix(img_dir + dirs + '/' + filename)

                    # Mise à une dimension
                    img = flatten_image(img)
                    data.append(img)

                    # Rajout du label
                    labels.append(dirs[dirs.find('-')+1:])

                    del img

    # Séparation des datasets testing/training
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)

    # Réduction de dimension
    pca = RandomizedPCA(n_components=10)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)

    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())

    knn = KNeighborsClassifier()
    knn.fit(train_x, train_y)

    test_y = np.array(test_y)
    train_y = np.array(train_y)

    res = pd.crosstab(test_y, knn.predict(test_x))
    res1 = res.values

    # Affichage des résultats
    print("Resultats :", round(100*res1.diagonal().sum()/len(test_y), 2), "%")
    for i in range(0, len(res)):
        print(res.index[i], ":", round(100*res1.diagonal()[i]/res.sum()[i], 2), "%")

#    knn.score(test_y, knn.predict(test_x))
