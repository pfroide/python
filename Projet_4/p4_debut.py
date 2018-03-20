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

    #data = data[data[nom_colonne] <= data[nom_colonne].quantile(0.98)]
    #data = data[data[nom_colonne2] <= data[nom_colonne2].quantile(0.98)]

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
    plt.xlim(limitemin,limitemax)
    # plt.hist(data[colon], bins=bin_values)
    # Test sans les valeurs NaN
    classes = np.linspace(-100, 100, 200)
    
    # Ligne rouge verticale
    plt.plot([0.0, 0], [0, 20000], 'r-', lw=2)
    
    # Données de l'histogramme
    plt.hist(data[colon][np.isfinite(data[colon])], bins=classes)
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

def f1(liste):
    compte = {}.fromkeys(set(liste),0)
    
    for valeur in liste:
        if valeur != 'nan':
            compte[valeur] += 1
    
    return compte

def relative_importance(data):
    from sklearn.ensemble import RandomForestRegressor
    
    # On récupère les features d'un côté...
    X = data.copy()
    X = X.drop(['ARR_DELAY'], axis=1) 
    
    #X = X.drop(['ORIGIN'], axis=1)
    #X = X.drop(['DEST'], axis=1)
    #X = X.drop(['FL_DATE'], axis=1)
    
    for p in X:
        if X[p].dtype == 'object':
            del X[p]
        elif 'DELAY' in p:
            del X[p]

    # TEST
    #X = X.drop(['DEP_DELAY'], axis=1)
    
    # On enlève les Nan
    X.fillna(0, inplace=True)
        
    # et les labels de l'autre
    y = data['ARR_DELAY'].copy()
    
    y.fillna(0, inplace=True)
    
    regr = RandomForestRegressor(max_depth=50, random_state=0)
    regr.fit(X, y)

    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=50,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)

    # Isolate feature importances
    # print(regr.feature_importances_)
    importance = regr.feature_importances_

    # Sort the feature importances
    sorted_importances = np.argsort(importance)

    # Insert padding
    names = X.columns.values[0:] 
    padding = np.arange(len(names)) + 0.5

    # Customize the plot
    plt.figure(figsize=(10,12))
    plt.yticks(padding, names[sorted_importances])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")

    # Plot the data
    plt.barh(padding, importance[sorted_importances], align='center')

    # Show the plot
    plt.show()
    
def regression(data_reg, colon1, colon2):
    
    # mask_colon2 = ~np.isnan(data_reg[colon2])
    mask_colon1 = np.isfinite(data_reg[colon1])
    mask_colon2 = np.isfinite(data_reg[colon2])
    mask = mask_colon1 & mask_colon2
    
    # Calcul d'une regression linéaire
    regr = linear_model.LinearRegression()
    
    # Reshape
    data_x = data_reg[colon1].values.reshape(-1, 1)
    data_y = data_reg[colon2].values.reshape(-1, 1)
    
    # Fit
    regr.fit(data_x[mask], data_y[mask])
    
    # Affichage de la variances : On doit être le plus proche possible de 1
    print('Regression sur :', colon1, colon2)
    print('Score : %.2f' % np.corrcoef(data_reg[colon1][mask], data_reg[colon2][mask])[1, 0])
    print('R2    : %.2f \n' % regr.score(data_x[mask], data_y[mask]))

def transpose_bool(data, colon, limite):
    """
    TBD
    """

    # On supprime les #NA
    data[colon].fillna('vide', inplace=True)

    # énumaration des genres
    listing = comptabiliser(data, colon)

    p = 0

    for mot, compte in listing:
        nom = colon + '_' + mot
        if p < limite:
            if nom not in data:
                data[nom] = pd.Series(((1 if mot in data[colon][i] else 0) for i in range(len(data[colon]))), index=data.index)
            else:
                data[nom] = pd.Series(((1 if (np.logical_or(data[nom].item == 1, mot in data[colon][i])) else 0) for i in range(len(data[colon]))), index=data.index)
        else:
            return p
        p = p+1

    return p

def svr(data):
        
    from sklearn.svm import SVR
    
    # #############################################################################
    # Generate sample data
    #X = np.sort(5 * np.random.rand(40, 1), axis=0)
    #y = np.sin(X).ravel()
    
    # #############################################################################
    # Add noise to targets
    #y[::5] += 3 * (0.5 - np.random.rand(8))
    
     # On récupère les features d'un côté...
    X = data.copy()
    X = X.drop(['ARR_DELAY'], axis=1) 
    
    X = X.drop(['ORIGIN'], axis=1)
    X = X.drop(['DEST'], axis=1)
    X = X.drop(['FL_DATE'], axis=1)
    
    # On enlève les Nan
    X.fillna(0, inplace=True)
        
    # et les labels de l'autre
    y = data['ARR_DELAY'].copy()
    y.fillna(0, inplace=True)
    
    # #############################################################################
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)
    
    # #############################################################################
    # Look at the results
    lw = 2
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
    plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

def main():
    """
    Fonction main
    """

    _FICHIERDATA = _DOSSIER + '2016_01.csv'
    
    # Récupération des dataset
    data = pd.read_csv(_FICHIERDATA, error_bad_lines=False)
    
    # Création du second dataset pour tester la popularité
    liste_criteres = []
    liste_criteres = ['FL_DATE', 'AIRLINE_ID', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID']
    liste_criteres.extend(['DEP_TIME', 'DEP_DELAY', 'ARR_TIME', 'ARR_DELAY'])
    liste_criteres.extend(['DISTANCE', 'AIR_TIME', 'LATE_AIRCRAFT_DELAY'])
    liste_criteres.extend(['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY'])
    liste_criteres.extend(['DAY_OF_WEEK', 'MONTH', 'ORIGIN', 'DEST'])
    liste_criteres.extend(['DAY_OF_MONTH', 'UNIQUE_CARRIER', 'DEP_TIME_BLK', 'ARR_TIME_BLK'])
    liste_criteres.extend(['ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'DISTANCE_GROUP'])
    
    # catégoriser les petits/grands retards
    # visualier les distributions des retards
    # régression linéaire multiple
    
    # On récupère toute les données
    data = pd.DataFrame({'A' : []})
    for i in range(1,13):
        if i <10:
            fichier = str('2016_0' + str(i) + '.csv')
        else:
            fichier = str('2016_' + str(i) + '.csv')
            
        datatemp = pd.read_csv(_DOSSIER + fichier, error_bad_lines=False)
        
        # Suppresion des données fausses
        datatemp=datatemp[datatemp['MONTH'] == i]
            
        data = pd.concat([data, datatemp])

    for colon in data:
            if colon not in liste_criteres:
                del data[colon]
    
    # Conversion en int de valeurs qui ne le seraient pas.
    data['DAY_OF_WEEK'] = data['DAY_OF_WEEK'].astype('int', copy=False)
    data['DAY_OF_MONTH'] = data['DAY_OF_MONTH'].astype('int', copy=False)    
        
    data = data.drop_duplicates(keep='first')
    
    data['Trajet'] = data['ORIGIN'] + ' to ' + data['DEST']
    
    tt_1 = data.groupby(['MONTH']).count()
    tt_2 = data.groupby(['ORIGIN']).count()
    tt_3 = data.groupby(['DEST']).count()
    tt_4 = data.groupby(['ARR_DELAY']).count()
    tt_5 = data.groupby(['AIRLINE_ID']).count()
    tt_6 = data[valeur_cherchee].value_counts()

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
            if 'DELAY' in nom_colonne:
                histogramme(data, nom_colonne, -70, 70)

    data = data[data['ARR_DELAY'] <= 60]
    data = data[data['ARR_DELAY'] >= -60]
    
    #for name in 'Trajet':
    res = comptabiliser(data, 'ARR_DELAY')
    afficher_plot('ARR_DELAY', res[0:100])

    # Pour le mois d'avril    
    res = comptabiliser(data, 'MONTH')
    afficher_plot('MONTH', res[0:50])
    
    scatter_plot(data, 'ARR_DELAY', 'DEP_DELAY')

    datanum = data.copy()
    datanum.describe()
    
    # Transposition en 0 et 1 des valeurs non-numériques
    liste_criteres = ['ORIGIN',
                      'DEST']

    for critere in liste_criteres:
        num = transpose_bool(datanum, critere, 50)
        print("Nombre : ", num, "\t", critere)
        
def graphique(data):

    liste = ['ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'MONTH', 'AIRLINE_ID', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'DEP_TIME_BLK', 'ARR_TIME_BLK', 'DISTANCE_GROUP']
    axe_y = 'ARR_DELAY'
    
    for classifieur in liste:

        # nom du fichier de sauvegarde
        fichier_save = _DOSSIERTRAVAIL + '\\' + classifieur
        
        # On range les données en faisaint la moyenne
        dep = data[[axe_y, classifieur]].groupby(classifieur, as_index=True).mean()
        
        # Tipe d'affichage et taille
        ax = dep.plot(kind = "bar", figsize=(len(dep)/2,6))
        
        # On fait les labels pour les afficher
        labels = ["%.2f" % i for i in dep[axe_y]]
        rects = ax.patches
        
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            # Différence entre chiffres négatifs et positifs
            if "-" not in label:
                ax.text(rect.get_x() + rect.get_width() / 2, height + 0.1, label, ha='center', va='bottom')
            else:
                ax.text(rect.get_x() + rect.get_width() / 2, height - 0.6, label, ha='center', va='bottom')
    
        # Titres
        ax.set_xlabel(classifieur, fontsize=10)
        ax.set_ylabel(axe_y, fontsize=10)
        Titre = "Retards pour " + classifieur
        ax.set_title(Titre, fontsize=16)
        
        # on supprime la légende
        ax.legend().set_visible(False)
        
        # Sauvegarde de la figure
        fig = ax.get_figure()
        fig.savefig(fichier_save, dpi=100)
