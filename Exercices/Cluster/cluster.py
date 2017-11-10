#! /usr/bin/env python3
# coding: utf-8

"""
Created on Mon Oct 30 11:27:30 2017

@author: Toni


Pour cet exercice il vous est demandé :

d’effectuer un partitionnement de X en 10 clusters, avec l’algorithme de clustering de votre choix
de visualiser le résultat de ce clustering en deux dimensions (obtenues par exemple grâce à tSNE)
d’évaluer la qualité de ce partitionnement, d’une part intrinsèquement (sans utiliser y) et d’autre part en le comparant aux chiffres représentés par les images (en utilisant y).

La matrice de confusion, dans la terminologie de l'apprentissage supervisé, est un outil servant à mesurer la qualité d'un système de classification.

Chaque colonne de la matrice représente le nombre d'occurrences d'une classe estimée, tandis que chaque ligne représente le nombre d'occurrences d'une classe réelle (ou de référence). Les données utilisées pour chacun de ces groupes doivent être différentes.

Un des intérêts de la matrice de confusion est qu'elle montre rapidement si le système parvient à classifier correctement.

"""

# Classification supervisée

from sklearn import preprocessing
from sklearn import cluster, metrics
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)
from sklearn.datasets import fetch_mldata
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from matplotlib import offsetbox

# Pour la séparation Training/Testing
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')

def main():
 
    # %pylab inline
    
    # Le dataset principal qui contient toutes les images
    print(mnist.data.shape)

    # Le vecteur d'annotations associé au dataset (nombre entre 0 et 9)
    print (mnist.target.shape)

    sample = np.random.randint(70000, size=25000)
    x_data = mnist.data[sample]
    y_data = mnist.target[sample]

    db_xtrain, db_xtest, db_ytrain, db_ytest = train_test_split(x_data, y_data, train_size=0.8)

    # Visualisation pour confirmation des shapes
    print("Shapes")
    print("db_xtrain", db_xtrain.shape)
    print("db_xtest", db_xtest.shape)
    print("db_ytrain", db_ytrain.shape)
    print("db_ytest", db_ytest.shape)

    # création de l'objet pca
    pca=decomposition.PCA()
    
    # application de l'objet
    pca.fit(db_xtrain)
    print(pca.explained_variance_ratio_.cumsum())
    X_trans=pca.transform(db_xtrain)
    
    cls=cluster.KMeans(n_clusters=10)
    cls.fit(X_trans)
    
    # taille des figures
    fig=plt.figure(figsize=(12,6))
    
    # figure 1
    ax=fig.add_subplot(121)
    ax.scatter(X_trans[:, 0], X_trans[:, 1], c=cls.labels_)
    plt.grid('on')
    
    # figure 2
    ax=fig.add_subplot(122)
    ax.scatter(X_trans[:, 0], X_trans[:, 1], c=db_ytrain)
    plt.grid('on')

    # On les montre
    plt.show()
    print("Valeur : %f " % metrics.adjusted_rand_score(cls.labels_,db_ytrain))

    # Matrice qui permets de voir les résultats trouvés en fonction 
    # des attendus
    cm = confusion_matrix(cls.labels_,db_ytrain)
    print(cm)

#    path="http://www.math.univ-toulouse.fr/~besse/Wikistat/data/"
#    Dtrain=pd.read_csv(path+"mnist_train.csv",header=None)
#    Dtrain.head()
#    
#    Ltrain=Dtrain.iloc[:,784]
#    Dtrain.drop(Dtrain.columns[[784]], axis=1,inplace=True)
#    Dtrain.shape
#    
#    Dtest=pd.read_csv(path+"mnist_test.csv",header=None)
#    Dtest.head()
#    Ltest=Dtest.iloc[:,784]
#    Dtest.drop(Dtest.columns[[784]], axis=1,inplace=True)
#    Dtest.shape
    
    # affichage d'un chiffre
#    plt.figure(1, figsize=(3, 3))
#    plt.imshow(np.matrix(db_xtrain).reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
#    plt.show()

    # Autre technique
    rf = RandomForestClassifier(n_estimators=100, 
                                criterion='gini', 
                                max_depth=None, 
                                min_samples_split=2,
                                min_samples_leaf=1, 
                                max_features='auto', 
                                max_leaf_nodes=None, 
                                bootstrap=True, 
                                oob_score=True, 
                                n_jobs=-1,
                                random_state=None, 
                                verbose=0)
    
    rf.fit(db_xtrain,db_ytrain)
    
    # erreur out-of-bag
    #print("Score:", rf.oob_score_)
    
    # erreur sur l'échantillon test
    print("Score", rf.score(db_xtest,db_ytest))
    
    # Matrice de confusion
    cm = confusion_matrix(db_ytest, rf.predict(db_xtest))
    print(cm)
    
    # Définition du modèle avec un nombre k "standard" de voisins
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=10,n_jobs=-1)
    digit_knn=knn.fit(db_xtrain, db_ytrain) 
    # Apprentissage et estimation de l'erreur de prévision sur l'échantillon test
    Score=digit_knn.score(db_xtest,db_ytest)
    print("Score : ",Score)

    cm = confusion_matrix(db_ytest, knn.predict(db_xtest))
    print(cm)
    
    
    
    for i in range(db_xtest.shape[0]):
        # Affichage
        plt.text(db_xtest[i], 
                 db_ytest[i], # à vous de définir ces dimensions !
                 str(db_ytest[i]),  # le point i est représenté par son chiffre
                 )
    
    
    pass

if __name__ == "__main__":
    main()
    

# fonction pour afficher une partie des images sur la visualisation 2D
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
            
def test():
    
    sample = np.random.randint(70000, size=50)
    data = mnist.data[sample]
    images = mnist.target[sample]
    
    X = data
    tsne = manifold.TSNE(n_components=2, perplexity=40, n_iter=3000, init='pca')
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne, "Principal Components projection of the digits")
    plt.show()
    
    # X = mnist.data[::50, :]
    # y = mnist.target[::50]

    """ 
    # determination du meilleur K
    silhouettes = []
 
    for num_clusters in range (9,11):
        cls=cluster.KMeans(n_clusters=num_clusters,
                           n_init=1,
                           init='random')
        cls.fit(X_norm)
        silh=metrics.silhouette_score(X_norm,cls.labels_)
        silhouettes.append(silh)
 
    plt.plot(range(9,11),silhouettes, marker='o')
    plt.show()
    
    
    #sample_idx = 42
    #sample_image = np.reshape(X[sample_idx, :], (28, 28))

    plt.imshow(sample_image, cmap='binary')
    """