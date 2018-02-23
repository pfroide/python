# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:22:16 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from matplotlib import pyplot as plt, cm as cm
from sklearn import linear_model

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\p4\\'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_4\\images'
_FICHIERDATA = _DOSSIER + '2016_01.csv'

def correlation_matrix(data):
    """
        Fonction qui permet de créer une visualisation du lien entre les
        variables 2 à 2
    """
    # Calcule de la matrice
    corr = data.corr()
    cmap = cm.get_cmap('jet', 30)

    # Taille de la figure
    plt.figure(figsize=(15, 15))
    # Création du type d'image
    cax = plt.imshow(data.corr(), interpolation="nearest", cmap=cmap)
    plt.grid(True)

    # Libellés sur les axes
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=10)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=10)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    plt.colorbar(cax, ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    plt.show()

def scatter_plot(data, nom_colonne2, nom_colonne):
    """
        Fonction qui permet d'afficher les nuages de points
    """

    #Log
    print("Fct affichage_plot\n")

    data = data[data[nom_colonne] <= data[nom_colonne].quantile(0.98)]
    data = data[data[nom_colonne2] <= data[nom_colonne2].quantile(0.98)]

    # Déliminations du visuel pour x
    xmax = max(data[nom_colonne])
    ymax = max(data[nom_colonne2])

    # Déliminations du visuel pour y
    xmin = min(data[nom_colonne])
    ymin = min(data[nom_colonne2])

    # création du nuage de point avec toujours la même ordonnée
    data.plot(kind="scatter", x=nom_colonne, y=nom_colonne2)

    # Affichage
    plt.grid(True)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()

def histogramme(data, colon):
    """
        Note : La première colonne et la dernière ont un " caché
    """

    #fichier_save = _DOSSIERTRAVAIL + '\\' + 'histogram_' + colon

    #steps = (max(data[colon])-min(data[colon]))/100
    #bin_values = np.arange(start=min(data[colon]), stop=max(data[colon]), step=steps)
    plt.figure(figsize=(10, 6))
    plt.xlabel('Valeurs')
    plt.ylabel('Décompte')
    titre = 'Histogramme ' + colon
    plt.title(titre)
    # plt.hist(data[colon], bins=bin_values)
    # Test sans les valeurs NaN
    plt.hist(data[colon][np.isfinite(data[colon])], bins=100)
    #plt.savefig(fichier_save, dpi=100)

def afficher_plot(type_donnee, trunc_occurences):
    """
    TBD
    """
    #fichier_save = _DOSSIERTRAVAIL + '\\' + type_donnee

    words = dict()

    for word in trunc_occurences:
        words[word[0]] = word[1]

    plt.figure(figsize=(15, 10))
    y_axis = [i[1] for i in trunc_occurences]
    x_axis = [k for k, i in enumerate(trunc_occurences)]
    
    x_label = [i[0] for i in trunc_occurences]
    
    plt.xticks(rotation=90, fontsize=10)
    plt.xticks(x_axis, x_label)

    plt.yticks(fontsize=10)
    plt.ylabel("Nb. of occurences", fontsize=18, labelpad=10)

    plt.bar(x_axis, y_axis, align='center', color='b')

    #plt.savefig(fichier_save, dpi=100)

    plt.title(type_donnee + " popularity", fontsize=25)
    plt.show()

def comptabiliser(data, valeur_cherchee):
    """
    TBD
    """
    # compter tous les genres différents
    listing = set()

    for word in data[valeur_cherchee].str.split('|').values:
        if isinstance(word, float):
            continue
        listing = listing.union(word)

    # compter le nombre d'occurence de ces genres
    listing_compte, dum = count_word(data, valeur_cherchee, listing)

    return listing_compte

def get_stats(param):
    """
    TBD
    """
    return {'min':param.min(),
            'max':param.max(),
            'count': param.count(),
            'mean':param.mean()
           }

def count_word(data, ref_col, liste):
    """
    TBD
    """
    keyword_count = dict()

    for word in liste:
        keyword_count[word] = 0

    for liste_keywords in data[ref_col].str.split('|'):
        if isinstance(liste_keywords, float) and pd.isnull(liste_keywords):
            continue
        for word in [word for word in liste_keywords if word in liste]:
            if pd.notnull(word):
                keyword_count[word] = keyword_count[word] + 1

    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []

    for k, v in keyword_count.items():
        keyword_occurences.append([k, v])

    keyword_occurences.sort(key=lambda x: x[1], reverse=True)

    return keyword_occurences, keyword_count
 
def main():
    """
    Fonction main
    """

    _FICHIERDATA = _DOSSIER + '2016_04.csv'

    _FICHIERDATA = 'C:\\Users\\Toni\\Desktop\\On_Time_On_Time_Performance_2016_4.csv'
    
    # Récupération des dataset
    data = pd.read_csv(_FICHIERDATA, error_bad_lines=False)
    
    # Création du second dataset pour tester la popularité
    liste_criteres = []
    liste_criteres = ['FL_DATE', 'AIRLINE_ID', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID']
    liste_criteres.extend(['DEP_TIME', 'DEP_DELAY', 'ARR_TIME', 'ARR_DELAY', 'CANCELLED'])
    liste_criteres.extend(['DISTANCE', 'AIR_TIME', 'LATE_AIRCRAFT_DELAY'])
    liste_criteres.extend(['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY'])
    liste_criteres.extend(['DAY_OF_WEEK', 'MONTH', 'ORIGIN', 'DEST'])
    
    data = pd.DataFrame({'A' : []})
    
    for i in range(1,13):
        if i <10:
            fichier = str('2016_0' + str(i) + '.csv')
        else:
            fichier = str('2016_' + str(i) + '.csv')
            
        datatemp = pd.read_csv(_DOSSIER + fichier, error_bad_lines=False)
        
        # Suppresion des données fausses
        datatemp=datatemp[datatemp['MONTH'] == i]
        
        for colon in datatemp:
            if colon not in liste_criteres:
                del datatemp[colon]
            
        data = pd.concat([data, datatemp])
        
    del data['Unnamed: 64']
    del data['A']
    
    result2 = result.copy()
    data = data.drop_duplicates(keep='first')
    
    data['Trajet'] = data['ORIGIN'] + ' to ' + data['DEST']
    
    p = data.groupby(['MONTH']).count()
    g = data.groupby(['ORIGIN']).count()
    m = data.groupby(['DEST']).count()
    b = data.groupby(['ARR_DELAY']).count()
    n = data.groupby(['Trajet']).count()
    
    #frames = [data, data2]
    #data = result.copy()

    # Données manquantes
    print("Données manquantes")
    missing_data = data.isnull().sum(axis=0).reset_index()
    missing_data.columns = ['column_name', 'missing_count']
    missing_data['filling_factor'] = (data.shape[0]-missing_data['missing_count'])/data.shape[0]*100
    missing_data.sort_values('filling_factor').reset_index(drop=True)

    # Transposition du dataframe de données pour l'analyse univariée
    fichier_save = _DOSSIERTRAVAIL + '\\' + 'transposition.csv'
    data_transpose = data.describe().reset_index().transpose()
    print(data_transpose)
    data_transpose.to_csv(fichier_save)
    
    correlation_matrix(data)

    # Création des histogrammes
    for nom_colonne in data:
        if data[nom_colonne].dtype == 'float' or data[nom_colonne].dtype == 'int64':
            histogramme(data, nom_colonne)

    #for name in 'Trajet':
    res = comptabiliser(data, 'Trajet')
    afficher_plot('Trajet', res[0:100])

    # Pour le mois d'avril    
    res = comptabiliser(data, 'MONTH')
    afficher_plot('MONTH', res[0:50])

    # 31 avril ?
    data['FL_DATE'] = data['FL_DATE'].str.replace('-03-', '-04-')