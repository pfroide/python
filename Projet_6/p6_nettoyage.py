# -*- coding: utf-8 -*-
"""
Created on Thu May 17 21:51:03 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import datetime
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from matplotlib import pyplot as plt, cm as cm
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p6\\'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_6\\images'

def suppr_code(fichier):
    """
    TBD
    """

    soupe = BeautifulSoup(fichier, "lxml")

    liste = soupe.findAll('code')

    for balise in liste:
        balise.decompose()

    return soupe

def parse_html(fichier_moche):
    """
    TBD
    """

    soupe = BeautifulSoup(fichier_moche, "lxml")

    return soupe.text

def donnees_manquantes(data, nom):
    """
    Données manquantes
    """

    # Données manquantes
    fichier_save = _DOSSIERTRAVAIL + '\\' + nom + '.csv'
    missing_data = data.isnull().sum(axis=0).reset_index()
    missing_data.columns = ['column_name', 'missing_count']
    missing_data['filling_factor'] = (data.shape[0]-missing_data['missing_count'])/data.shape[0]*100
    print(missing_data.sort_values('filling_factor').reset_index(drop=True))
    missing_data.sort_values('filling_factor').reset_index(drop=True).to_csv(fichier_save)

    # Transposition du dataframe de données pour l'analyse univariée
    fichier_save = _DOSSIERTRAVAIL + '\\' + 'transposition.csv'
    data_transpose = data.describe().reset_index().transpose()
    print(data_transpose)
    data_transpose.to_csv(fichier_save)

def main():
    """
    Fonction principale
    """

    # Récupération du dataset
    fichier = 'QueryResults.csv'
    data = pd.read_csv(_DOSSIER + fichier, error_bad_lines=False)

    data = data[0:500]

    # Données manquantes
    donnees_manquantes(data, "missing_data_1")

    #for ligne in range(0, len(data)):
    #    data.loc[ligne, 'Body'] = parse_html(data.loc[ligne, 'Body'])
    data['Body'] = [parse_html(x) for x in data['Body']]
    data['Body'] = [suppr_code(x) for x in data['Body']]
