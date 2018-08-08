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
STANDARD_SIZE = (500, 375)

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

def classification_retards(data):
    """
    TBD
    """

    # Classification des retards
    for dataset in data:
        data.loc[data['ARR_DELAY'] <= 15, 'CLASSE_DELAY'] = "Leger"
        data.loc[data['ARR_DELAY'] >= 15, 'CLASSE_DELAY'] = "Moyen"
        data.loc[data['ARR_DELAY'] >= 45, 'CLASSE_DELAY'] = "Important"
        data.loc[data['ARR_DELAY'] < 0, 'CLASSE_DELAY'] = "En avance"

    # Affichage de la classification des retards
    f, ax = plt.subplots(1, 2, figsize=(20, 8))
    data['CLASSE_DELAY'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title('CLASSE_DELAY')
    ax[0].set_ylabel('')
    sns.countplot('CLASSE_DELAY',order=data['CLASSE_DELAY'].value_counts().index, data=data, ax=ax[1])
    ax[1].set_title('Status')
    plt.tight_layout()
    #plt.savefig(_DOSSIERTRAVAIL + '\\' + 'classification_retards', dpi=100)
    plt.show()

    del data['CLASSE_DELAY']

def main():
    """
    TBD
    """

    img_dir = '/home/toni/Bureau/p7/Images/' # n02090379-redbone/'
    #images = [img_dir+ f for f in os.listdir(img_dir)]
    #labels = ["check" if "check" in f.split('/')[-1] else "redbone" for f in images]

    data = []
    labels = []
    spec_images = pd.DataFrame()

    i=0

    from scipy import misc

    for path, dirs, files in os.walk(img_dir):
        for filename in files:
            M = misc.imread(path + '/' + filename)
            titre = path + '/' + filename
            spec_images.loc[titre, 0] = M.shape[0]
            spec_images.loc[titre, 1] = M.shape[1]
            spec_images.loc[titre, 2] = M.shape[2]
            spec_images.loc[titre, 3] = str(M.shape[0]) + '-' + str(M.shape[1]) + '-' + str(M.shape[2])

    df = spec_images.groupby(3)[0].count()

    import matplotlib.pyplot as plt
    plt.pie(df, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.show()

    for path, dirs, files in os.walk(img_dir):
        i=0
        for filename in files:
            if i<10:
                i=i+1
                if i%100==0:
                    #print(filename)
                    print(i)
                img = img_to_matrix(path + '/' + filename)
                img = flatten_image(img)
                data.append(img)
                #labels.append(path)
                labels.append(path[path.find('-')+1:])

    #data = np.array(data).astype(int)
    data

    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)

    pca = RandomizedPCA(n_components=10)
    train_x = pca.fit_transform(train_x)
    print(pca.explained_variance_ratio_ )
    print(pca.explained_variance_ratio_.sum())
    test_x = pca.transform(test_x)
    knn = KNeighborsClassifier()
    knn.fit(train_x, train_y)

    test_y = np.array(test_y)
    train_y = np.array(train_y)

    res = pd.crosstab(test_y, knn.predict(test_x))
    knn.score(test_y, knn.predict(test_x))
