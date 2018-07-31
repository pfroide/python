#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 21:38:27 2018

@author: toni
"""

import scipy.io
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Import to a python dictionary
mat = scipy.io.loadmat('/home/toni/Bureau/p7/test_data.mat')

# Look at the dictionary items
mat.items()

# Print the data
m = mat["test_data"]


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
from PIL import Image
import numpy as np

#setup a standard image size; this will distort some images but will get everything into the same shape
STANDARD_SIZE = (300, 167)

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

    img_dir = '/home/toni/Bureau/p7/Images/' # n02090379-redbone/'
    images = [img_dir+ f for f in os.listdir(img_dir)]
    #labels = ["check" if "check" in f.split('/')[-1] else "redbone" for f in images]

    data = []
    labels = []
    i=0

    for path, dirs, files in os.walk(img_dir):
        for filename in files:
            if i<1000:
                i=i+1
                print(filename)
                img = img_to_matrix(path + '/' + filename)
                img = flatten_image(img)
                data.append(img)
                labels.append(path)

#    for image in images:
#        img = img_to_matrix(image)
#        img = flatten_image(img)
#        data.append(img)

    data = np.array(data)
    data

    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25, random_state=42)

    pca = RandomizedPCA(n_components=10)
    train_x = pca.fit_transform(train_x)
    print(pca.explained_variance_ratio_ )
    test_x = pca.transform(test_x)
    knn = KNeighborsClassifier()
    knn.fit(train_x, train_y)

    test_y = np.array(test_y)
    train_y = np.array(train_y)

    res = pd.crosstab(test_y, knn.predict(test_x))
    knn.score(test_y, knn.predict(test_x))
