#! /usr/bin/env python3
# coding: utf-8

"""
Created on Mon Oct 30 11:27:30 2017

Pour cet exercice il nous est demandé :
PARTIE 1 Effectuer un partitionnement de X en 10 clusters, avec l’algorithme
de clustering de notre choix.

PARTIE 2 Visualiser le résultat de ce clustering en deux dimensions.

PARTIE 3 Evaluer la qualité de ce partitionnement, d’une part intrinsèquement
(sans utiliser y) et d’autre part en le comparant aux chiffres représentés
par les images (en utilisant y).
"""

# Classification supervisée

#from sklearn import preprocessing
from sklearn import cluster, metrics, manifold
from sklearn.datasets import fetch_mldata
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Pour la séparation Training/Testing
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

data_mnist = fetch_mldata('MNIST original')

def main():
    """
        Main avec toutes les opérations à la suite
    """

    # Le dataset principal qui contient toutes les images
    print(data_mnist.data.shape)

    # Le vecteur d'annotations associé au dataset (nombre entre 0 et 9)
    print(data_mnist.target.shape)

    sample = np.random.randint(70000, size=2500)
    x_data = data_mnist.data[sample]
    y_data = data_mnist.target[sample]

    db_xtrain, db_xtest, db_ytrain, db_ytest = train_test_split(x_data, y_data, train_size=0.8)

    # Visualisation pour confirmation des shapes
    print("Shapes")
    print("db_xtrain", db_xtrain.shape)
    print("db_xtest", db_xtest.shape)
    print("db_ytrain", db_ytrain.shape)
    print("db_ytest", db_ytest.shape)

    # création de l'objet
    cls = cluster.KMeans(n_init=50,
                         n_clusters=10,
                         init="k-means++",
                         max_iter=1000,
                         verbose=1,
                         algorithm="auto")

    # application de l'objet sur les données
    cls.fit(x_data)

    # On va essayer maintenant l'algorithme t-SNE, pour
    # “t-distributed Stochastic Neighbor Embedding”. Cette technique permet
    # de visualiser des données de grandes dimensions dans une variété de
    # plus petite dimension. Nous allons essayer avec deux dimensions,
    # ce qui nous permetterait de visualiser les résultats directement
    # dans un graphique.
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, perplexity=40, n_iter=3000)

    # perform t-SNE embedding
    x_tsne = tsne.fit_transform(x_data)

    # plot the result
    vis_x = x_tsne[:, 0]
    vis_y = x_tsne[:, 1]

    # figure de prédiction
    plt.scatter(vis_x, vis_y, c=cls.labels_, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.grid('on')
    plt.title("Affichage du clustering déduit par le Kmeans")
    plt.show()

    # figure avec les données réelles
    plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.grid('on')
    plt.title("Affichage du clustering réel")
    plt.show()

    # Visualisation du score
    print(metrics.adjusted_rand_score(y_data, cls.labels_))

    # La matrice de confusion, dans la terminologie de l'apprentissage
    # supervisé, est un outil servant à mesurer la qualité d'un système
    # de classification.
    # Chaque colonne de la matrice représente le nombre d'occurrences d'une
    # classe estimée, tandis que chaque ligne représente le nombre d'occurrences
    # d'une classe réelle (ou de référence). Les données utilisées pour chacun
    # de ces groupes doivent être différentes.
    # Un des intérêts de la matrice de confusion est qu'elle montre rapidement
    # si le système parvient à classifier correctement.
    # Affichage de la matrice de confusion
    print(confusion_matrix(cls.labels_, y_data))

    # Autre technique de prédiction/classification
    rf_object = RandomForestClassifier(n_estimators=100,
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

    # Fit des données
    rf_object.fit(db_xtrain, db_ytrain)

    # erreur sur l'échantillon test
    print(" \nErreur du rf : ", 1-rf_object.score(db_xtest, db_ytest))

    # Matrice de confusion
    print(confusion_matrix(db_ytest, rf_object.predict(db_xtest)))

    # Création de l'objet
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, perplexity=40, n_iter=3000)

    # perform t-SNE embedding
    x_tsne = tsne.fit_transform(db_xtest)

    # Séparation des deux composantes issues de la tsne
    vis_x = x_tsne[:, 0]
    vis_y = x_tsne[:, 1]

    # figure de prédiction
    plt.scatter(vis_x, vis_y, c=rf_object.predict(db_xtest), cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.grid('on')
    plt.title("Affichage du clustering déduit par le tsne")
    plt.show()

    # figure avec les données réelles
    plt.scatter(vis_x, vis_y, c=db_ytest, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.grid('on')
    plt.title("Affichage du clustering réel")
    plt.show()

    # Affichage de la matrice de confusion
    print(confusion_matrix(db_ytest, rf_object.predict(db_xtest)))

    # Le k-NN
    # On peut créer un premier classifieur 3-NN, c'est à dire qui prend
    # en compte les 3 plus proches voisins pour la classification.

    # Définition du modèle avec un nombre k "standard" de voisins
    knn = KNeighborsClassifier(n_neighbors=10)

    # fit des données sur l'objet
    digit_knn = knn.fit(db_xtrain, db_ytrain)

    # Comme je l'ai dit plus haut pour le k-NN, l'algorithme ici n'effectue
    # aucune optimisation,
    # mais va juste sauvegarder toutes les données en mémoire.
    # C'est sa manière d'apprendre en quelque sorte.
    # Apprentissage et estimation de l'erreur de prévision sur l'échantillon test
    print("\nScore du knn : ", digit_knn.score(db_xtest, db_ytest))

    # On récupère les prédictions sur les données test
    predicted = knn.predict(db_xtest)

    # On redimensionne les données sous forme d'images
    images = db_xtest.reshape((-1, 28, 28))

    # On selectionne un echantillon de 12 images au hasard
    select = np.random.randint(images.shape[0], size=12)

    # On affiche les images avec la prédiction associée
    for index, value in enumerate(select):
        plt.subplot(3, 4, index+1)
        plt.axis('off')
        plt.imshow(images[value], cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title('Predicted: %i' % predicted[value])

    plt.show()

    # Pour pouvoir un peu mieux comprendre les erreurs effectuées par le classifieur,
    # on peut aussi afficher un extrait des prédictions erronées :

    # on récupère les données mal prédites
    misclass = (db_ytest != predicted)
    misclass_images = images[misclass, :, :]
    misclass_predicted = predicted[misclass]

    # on sélectionne un échantillon de ces images
    select = np.random.randint(misclass_images.shape[0], size=12)

    # on affiche les images et les prédictions (erronées) associées à ces images
    for index, value in enumerate(select):
        plt.subplot(3, 4, index+1)
        plt.axis('off')
        plt.imshow(misclass_images[value], cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title('Predicted: %i' % misclass_predicted[value])

    plt.show()

    # Affichage de la matrice de confusion
    print(confusion_matrix(db_ytest, knn.predict(db_xtest)))

    # figure de prédiction
    plt.scatter(vis_x, vis_y, c=knn.predict(db_xtest), cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.grid('on')
    plt.title("Affichage du clustering déduit par le knn")
    plt.show()

    # figure avec les données réelles
    plt.scatter(vis_x, vis_y, c=db_ytest, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.grid('on')
    plt.title("Affichage du clustering réel")
    plt.show()

if __name__ == "__main__":
    main()
