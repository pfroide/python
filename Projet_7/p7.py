#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 21:38:27 2018

@author: toni
"""

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

import os
import cv2
import random
import warnings
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.cluster.vq import whiten
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

# Pour ne pas avoir les warnings lors de la compilation
warnings.filterwarnings("ignore")

# Lieu où se trouvent des images
IMG_DIR = '/home/toni/Bureau/p7/Images/'

# Définitions des limites d'execution
NB_RACES = 10
NB_EXEMPLES = 10

#setup a standard image size; this will distort images but will get everything into the same shape
STANDARD_SIZE = (300, 167)
#STANDARD_SIZE = (500, 375)

def fonction_median(img, param1):
    """
    Fonction de filtre
    """

    # Application du filtre
    img_modified = scipy.ndimage.median_filter(img, size=param1)

    return img_modified

def fonction_gauss(img, param1):
    """
    Fonction de filtre
    """

    # Application du filtre
    img_modified = scipy.ndimage.filters.gaussian_filter(img, sigma=param1)

    return img_modified

def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """

    img = Image.open(filename)

    if verbose:
        print("changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE)))

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

    shape = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, shape)
    return img_wide[0]

def recup_images(liste_dossier, num_filtre):
    """
    Fonction qui récupére toute les images avec une sélection aléatoire
    Rajout de filtres possibles
    """

    # Création des listes vides
    data = []
    labels = []

    # Valeur initiale d'un compteur
    cpt_race = 0

    #
    orb = cv2.ORB_create()

    for dirs in liste_dossier:
        # Valeur initiale d'un compteur
        cpt_exemple = 0
        if cpt_race < NB_RACES+1:
            cpt_race = cpt_race+1
            for filename in os.listdir(IMG_DIR + dirs):
                # On ne garde que NB_EXEMPLES exemplaires de chaque race
                if cpt_exemple < NB_EXEMPLES:
                    cpt_exemple = cpt_exemple+1

                    # Affichage pour voir si ça tourne toujours
                    #if cpt_exemple%25 == 0:
                    #    print(cpt_exemple, dirs + '/' + filename)

                    # Récupération de la matrice tranformée
                    img = img_to_matrix(IMG_DIR + dirs + '/' + filename, False)

                    if num_filtre == 1:
                        # Filtre gaussien
                        img = fonction_gauss(img, 5)
                    elif num_filtre == 2:
                        # Filtre médian
                        img = fonction_median(img, 5)
                    elif num_filtre == 3:
                        img = whiten(img)
                    elif num_filtre == 4:
                        image_brute = IMG_DIR + dirs + '/' + filename
                        a, b = image_detect_and_compute(orb, image_brute)
                        data.append(b)

                    # Mise à une dimension
                    #img = flatten_image(img)
                    #data.append(img)

                    # Rajout du label
                    labels.append(dirs[dirs.find('-')+1:].lower())

                    del img

    return data, labels

def image_detect_and_compute(detector, img):
    """
    Detect and compute interest points and their descriptors.
    """

    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(img, None)

    return kp, des

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
        #print(res.index[i], ":", round(100*res1.diagonal()[i]/res.sum()[i], 2), "%")

    data_resultats = data_resultats.fillna(0)

    print(data_resultats)

def main():
    """
    Fonction principale
    """

    liste_dossier = []

    # Création de la liste aléatoire des races
    liste_chiens = os.listdir(IMG_DIR)
    for i in range(0, NB_RACES):
        nb_alea = random.randrange(0, len(liste_chiens))
        liste_dossier.append(liste_chiens[nb_alea])
        del liste_chiens[nb_alea]

    ## Différents filtres
    for filtre in range(4, 5):
        # FILTRE 0 - AUCUN
        # FILTRE 1- GAUSSIEN
        # FILTRE 2 - MEDIAN
        if filtre == 0:
            nom_filtre = "Aucun"
        elif filtre == 1:
            nom_filtre = "Gaussien"
        elif filtre == 2:
            nom_filtre = "Median"
        elif filtre == 3:
            nom_filtre = "Whitening"
        elif filtre == 4:
            nom_filtre = "ORB"

        print("\nFiltre", nom_filtre)
        data, labels = recup_images(liste_dossier, filtre)

        ## Réduction de dimension
        # PCA
        pca = RandomizedPCA(n_components=10)
        data = pca.fit_transform(data)
        # Explication de la variance
        print(pca.explained_variance_ratio_.sum())

        # t-SNE
        #data = TSNE(n_components=2).fit_transform(data, labels)

        # Séparation des datasets testing/training
        train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)

        # Transformation en array
        test_y = np.array(test_y)
        train_y = np.array(train_y)

        ## Création de la méthode de classification
        # Test avec KNN
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(train_x, train_y)
        res = pd.crosstab(test_y, knn.predict(test_x))
        # Gestion d'une erreur
        if len(res.columns) != NB_RACES:
            res = gestion_erreur(res, test_y, '0', 'knn')
        calcul_resultats(res, test_y, 'knn')

        # Test avec Kmeans
        kmeans = KMeans(n_clusters=NB_RACES).fit(train_x, train_y)
        res = pd.crosstab(test_y, kmeans.predict(test_x))
        # Gestion d'une erreur
        if len(res.columns) != NB_RACES:
            res = gestion_erreur(res, test_y, labels, 'kmeans')
        calcul_resultats(res, test_y, 'kmeans')

def gestion_erreur(res, test_y, labels, classifieur):
    """
    Gestion de l'erreur quand une catégorie de chien n'est pas prédite
    On rajoute la colonne vide manuellement
    """

    if classifieur is not 'kmeans':
        for i in np.unique(test_y):
            if i not in res.columns:
                res[i] = 0

        for i in res.columns:
            if i not in res.index:
                res.loc[i] = 0
    else:
        for i in range(0, NB_RACES):
            if i not in res.columns:
                res[i] = 0

        for i in np.unique(labels):
            if i not in res.index:
                res.loc[i] = 0

    res = res.sort_index(axis=0, ascending=True)
    res = res.sort_index(axis=1, ascending=True)

    return res

#-----
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
