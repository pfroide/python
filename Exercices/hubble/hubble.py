# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:39:21 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
import os

# Lieu où se trouve le FICHIER
_FICHIER = os.getcwd() + '\\Desktop\\hubble_data.csv'

def histogramme(data, colon):
    """
        Note : La première colonne et la dernière ont un " caché
    """

    steps = (max(data[colon])-min(data[colon]))/20
    bin_values = np.arange(start=min(data[colon]), stop=max(data[colon]), step=steps)
    plt.figure(figsize=(10, 6))
    plt.xlabel('Valeurs')
    plt.ylabel('Décompte')
    titre = 'Histogramme ' + colon
    plt.title(titre)
    plt.hist(data[colon], bins=bin_values)
    
def regression_lineaire(data, colon1, colon2):
    """
        Fonction pour le calcul des régressions linéaires
    """

    # Calcul de la droite optimale
    regr = linear_model.LinearRegression()
    regr.fit(data[colon1].values.reshape(-1, 1), data[colon2].values.reshape(-1, 1))
    score = np.corrcoef(data[colon1], data[colon2])[1, 0]

    # Affichage de la variances : On doit être le plus proche possible de 1
    print('Régression sur les deux données :', colon1, "et", colon2)
    print('Score : %.2f' % score)

    # Affichage de la droite optimale)
    plt.figure(figsize=(10, 6))
    plt.plot([0, 2], [regr.intercept_, regr.intercept_ + 2*regr.coef_], linewidth=2.0, label="droite de régression")
    plt.plot(data[colon1], data[colon2],'ro', markersize=3, label="points de données")
    plt.title('Vitesse de récession en fonction de la distance')
    plt.xlabel('Distance')
    plt.ylabel('Vitesse')
    plt.grid('on')
    plt.legend()
    plt.show()
    
def main():
    
    # On charge le dataset sur les colonnes qui nous ont intéressés dans la
    # fonction du dessus
    data = pd.read_csv(_FICHIER,
                       error_bad_lines=False,
                       engine='python',
                       sep=',')
    
    # données brutes 
    print(data)
    
    # Visualisation des données
    for colon in data:
        histogramme(data, colon)
    
    # Transposition du dataframe de données pour l'analyse univariée
    data_transpose = data.describe().reset_index().transpose()
    print (data_transpose)
    
    regression_lineaire(data, 'distance', 'recession_velocity')
    