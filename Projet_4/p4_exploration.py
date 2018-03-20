# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 20:33:54 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm as cm
from sklearn.ensemble import RandomForestRegressor

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\p4\\'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_4\\images'
_FICHIERDATA = _DOSSIER + '2016_01.csv'

def retard_par_aeroport(data):
    """
    TBD
    """

    airport_mean_delays = pd.DataFrame(pd.Series(data['ORIGIN_AIRPORT_ID'].unique()))

    airport_mean_delays.set_index(0, drop=True, inplace=True)

    airport_mean_delays[airport_mean_delays.index.duplicated()]

    for carrier in data['UNIQUE_CARRIER'].unique():
        df1 = data[data['UNIQUE_CARRIER'] == carrier]
        test = df1['ARR_DELAY'].groupby(data['ORIGIN_AIRPORT_ID']).apply(get_stats).unstack()
        airport_mean_delays[carrier] = test.loc[:, 'mean']

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

def histogramme(data, colon, limitemin, limitemax):
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
    plt.xlim(limitemin, limitemax)
    # plt.hist(data[colon], bins=bin_values)
    # Test sans les valeurs NaN
    classes = np.linspace(-100, 100, 200)

    # Ligne rouge verticale
    plt.plot([0.0, 0], [0, 2000], 'r-', lw=2)

    # Données de l'histogramme
    plt.hist(data[colon][np.isfinite(data[colon])], bins=classes)
    #plt.savefig(fichier_save, dpi=100)

def get_stats(param):
    """
    TBD
    """
    return {'min':param.min(),
            'max':param.max(),
            'count': param.count(),
            'mean':param.mean()
           }

def relative_importance(data):
    """
    TBD
    """

    # On récupère les features d'un côté...
    data_x = data.copy()
    data_x = data_x.drop(['ARR_DELAY'], axis=1)

    for colon in data_x:
        if data_x[colon].dtype == 'object':
            del data_x[colon]
        elif 'DELAY' in colon:
            del data_x[colon]

    # On enlève les Nan
    data_x.fillna(0, inplace=True)

    # et les labels de l'autre
    data_y = data['ARR_DELAY'].copy()

    data_y.fillna(0, inplace=True)

    regr = RandomForestRegressor(max_depth=50, random_state=0)
    regr.fit(data_x, data_y)

    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=50,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                          oob_score=False, random_state=0, verbose=0, warm_start=False)

    # Isolate feature importances
    importance = regr.feature_importances_

    # Sort the feature importances
    sorted_importances = np.argsort(importance)

    # Insert padding
    names = data_x.columns.values[0:]
    padding = np.arange(len(names)) + 0.5

    # Customize the plot
    plt.figure(figsize=(10, 12))
    plt.yticks(padding, names[sorted_importances])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")

    # Plot the data
    plt.barh(padding, importance[sorted_importances], align='center')

    # Show the plot
    plt.show()

def graphique_par_donnee(data, classifieur):
    """
    TBD
    """

    axe_y = 'ARR_DELAY'

    # nom du fichier de sauvegarde
    fichier_save = _DOSSIERTRAVAIL + '\\' + classifieur

    # On range les données en faisaint la moyenne
    dep = data[[axe_y, classifieur]].groupby(classifieur, as_index=True).mean()

    # Tipe d'affichage et taille
    axe = dep.plot(kind="bar", figsize=(len(dep)/2, 6))

    # On fait les labels pour les afficher
    labels = ["%.2f" % i for i in dep[axe_y]]
    rects = axe.patches

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        width = rect.get_width()

        # Différence entre chiffres négatifs et positifs
        if "-" not in label:
            axe.text(rect.get_x() + width / 2, height + 0.1, label, ha='center', va='bottom')
        else:
            axe.text(rect.get_x() + width / 2, height - 0.6, label, ha='center', va='bottom')

    # Titres
    axe.set_xlabel(classifieur, fontsize=10)
    axe.set_ylabel(axe_y, fontsize=10)
    titre = "Retards pour " + classifieur
    axe.set_title(titre, fontsize=16)

    # on supprime la légende
    axe.legend().set_visible(False)

    # Sauvegarde de la figure
    fig = axe.get_figure()
    fig.savefig(fichier_save, dpi=100)

def main():
    """
    Fonction main
    """

    # Récupération des dataset
    # Pour un mois
    data = pd.read_csv(_FICHIERDATA, error_bad_lines=False, low_memory=False)

    # Pour toute l'année
    data = pd.DataFrame({'A' : []})
    for i in range(1, 13):
        if i < 10:
            fichier = str('2016_0' + str(i) + '.csv')
        else:
            fichier = str('2016_' + str(i) + '.csv')

        datatemp = pd.read_csv(_DOSSIER + fichier, error_bad_lines=False, low_memory=False)

        # Suppresion des données fausses
        datatemp = datatemp[datatemp['MONTH'] == i]

        data = pd.concat([data, datatemp])

    # Conversion en int de valeurs qui ne le seraient pas.
    data['DAY_OF_WEEK'] = data['DAY_OF_WEEK'].astype('int', copy=False)
    data['DAY_OF_MONTH'] = data['DAY_OF_MONTH'].astype('int', copy=False)
    data['ORIGIN_AIRPORT_ID'] = data['ORIGIN_AIRPORT_ID'].astype('float', copy=False)
    data['ORIGIN_AIRPORT_ID'] = data['ORIGIN_AIRPORT_ID'].astype('int', copy=False)

    # Liste des critères à conserver
    liste_criteres = []
    liste_criteres = ['FL_DATE', 'AIRLINE_ID', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID']
    liste_criteres.extend(['DEP_TIME', 'DEP_DELAY', 'ARR_TIME', 'ARR_DELAY'])
    liste_criteres.extend(['DISTANCE', 'AIR_TIME', 'LATE_AIRCRAFT_DELAY'])
    liste_criteres.extend(['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY'])
    liste_criteres.extend(['DAY_OF_WEEK', 'MONTH', 'ORIGIN', 'DEST'])
    liste_criteres.extend(['DAY_OF_MONTH', 'UNIQUE_CARRIER', 'DEP_TIME_BLK', 'ARR_TIME_BLK'])
    liste_criteres.extend(['ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'DISTANCE_GROUP'])

    for colon in data:
        if colon not in liste_criteres:
            del data[colon]

    data = data.drop_duplicates(keep='first')

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

    # Affichage
    liste_affichage = []
    liste_affichage = ['ORIGIN_AIRPORT_ID', 'ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'MONTH']
    liste_affichage.extend(['AIRLINE_ID', 'DAY_OF_MONTH', 'DAY_OF_WEEK'])
    liste_affichage.extend(['UNIQUE_CARRIER', 'DEP_TIME_BLK', 'ARR_TIME_BLK', 'DISTANCE_GROUP'])

    for colon in liste_affichage:
        graphique_par_donnee(data, colon)

    for colon in liste_affichage:
        print("Nb of", colon, " : ", len(data[colon].unique()))

    # Création des histogrammes
    for nom_colonne in data:
        if data[nom_colonne].dtype == 'float' or data[nom_colonne].dtype == 'int64':
            if 'DELAY' in nom_colonne:
                histogramme(data, nom_colonne, -70, 70)

    retard_par_aeroport(data)

#    tt_1 = data.groupby(['MONTH']).count()
#    tt_2 = data.groupby(['ORIGIN']).count()
#    tt_3 = data.groupby(['DEST']).count()
#    tt_4 = data.groupby(['ARR_DELAY']).count()
#    tt_5 = data.groupby(['AIRLINE_ID']).count()
