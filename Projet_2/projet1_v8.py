"""
    Projet n°2.
    OpenFood
"""
#! /usr/bin/env python3
# coding: utf-8

# On importe les librairies dont on aura besoin pour ce tp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn import decomposition

# Valeur limite des Nan acceptée
# Il faut donc moins de _VALEUR_LIMITE_NAN valeurs "NaN" pour garder la colonne
_VALEUR_LIMITE_NAN = 230000

# Référence en y du plot
_ORDONNEE = "nutrition-score-fr_100g"

# Lieu où se trouve le FICHIER
_FICHIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\bdd.csv'

# que faire des données manquantes ?
# affichage avant/après traitements
# faire un modèle de prédiction du score nutrionnel (continue/classe)

def definir_importance(data):
    """
        Fonction qui permets d'afficher un graphique
        où l'on verra l'importance relative de chaque élément dans le calcul
        final
    """

    # Log
    print("Fct definir_importance : \n")

    # Trouver le numéro de colonne qui nous sert d'ordonné dans l'affichage
    position_ordonne = data.columns.get_loc(_ORDONNEE)

    # Isolate Data, class labels and column values
    #X = data.iloc[:,0:15]       # colonne 0 à 15
    xdata = data.iloc[:, 0:position_ordonne]    # toutes les colonnes sauf la dernière
    ydata = data.iloc[:, position_ordonne]      # dernière colonne

    # names = data.columns.values[0:15]                 # avec limite
    names = data.columns.values[0:position_ordonne]     # sans limite

    # Build the model
    rfc = RandomForestClassifier(criterion='gini', verbose=1)

    # Fit the model
    rfc.fit(xdata, ydata)

    # Print the results
    #print("Features sorted by their score:")
    #print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), names), reverse = True))

    # Isolate feature importances
    importance = rfc.feature_importances_

    # Sort the feature importances
    sorted_importances = np.argsort(importance)

    # Insert padding
    padding = np.arange(len(names)) + 0.5

    # Customize the plot
    plt.yticks(padding, names[sorted_importances])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")

    # Plot the data
    plt.barh(padding, importance[sorted_importances], align='center')

    # Show the plot
    plt.show()

def affichage_plot(data, nom_colonne):
    """
        Fonction qui permets d'afficher les nuages de points
    """

    #Log
    print("Fct affichage_plot : Affichage de la courbe\n")

    # Déliminations du visuel pour x
    xmax = max(data[nom_colonne])
    ymax = max(data[_ORDONNEE])

    # Déliminations du visuel pour y
    xmin = min(data[nom_colonne])
    ymin = min(data[_ORDONNEE])

    # création du nuage de point avec toujours la même ordonnée
    data.plot(kind="scatter", x=nom_colonne, y=_ORDONNEE)

    # Affichage
    plt.grid(True)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()

def remplir_colonnes(data, nom_colonne, colonnes):
    """
        Fonction qui permets de selectionner des colonnes pour la
        base de données
    """

    # Log
    print("\nFct remplir_colonnes : Traitement de : %s" % nom_colonne)

    # test du type de la colonne. IL n'y a que les valeurs numériques qui
    # nous intéressent
    if data[nom_colonne].dtype == 'float':
        # si "100g" est trouvé dans le nom de la colonne
        if nom_colonne.find('100g') != -1 and nom_colonne.find('uk') == -1:
            colonnes.append(nom_colonne)
            print("Cette donnée est gardée")
        else:
            print("Cette donnée est exclue : pas de 100g")

    else:
        print("Cette donnée est exclue : pas un float")

def supprimer_colonnes(data, nom_colonne):
    """
        Fonction qui permets de supprimer des colonnes de la bdd
    """

    # Log
    print("\nFct supprimer_colonnes : Traitement de : %s" % nom_colonne)

    # nombre de valeurs "NaN"
    # .isnull().sum() = nombre par ligne
    # .isnull().sum().sum() = nombre total
    cpt_nan = data[nom_colonne].isnull().sum().sum()

    # S'il y a plus de valeur "Nan" que le chiffre défini, on vire la colonne
    if cpt_nan > (_VALEUR_LIMITE_NAN):
        # Suprresion de la colonne
        del data[nom_colonne]

        # Log
        print("Cette donnée est exclue : elle contient %.0f 'NaN' " % cpt_nan)
    else:
        # Log
        print("Cette donnée est gardée : elle contient %.0f 'NaN' " % cpt_nan)

def main():
    """
        Note : La première colonne et la dernière ont un " caché
        NB1 :
            J'ai supprimé la dernière sans vérifier qu'elle était bonne
        NB2 :
            Si on additione tous les ingrédients, et c'est supérieur à 100g,
            il faudra sans doute supprimer la colonne
    """

    # Définition de la variable qui récupère le nom des colonnes
    colonnes = []

    # On charge la première ligne du dataset
    bdd_titres = pd.read_csv(_FICHIER,
                             nrows=1,
                             error_bad_lines=False,
                             engine='python',
                             sep=r'\t')

    # Fonction qui va choisir les colonnes à récupérer suivant les crières
    # définis
    for i in bdd_titres:
        remplir_colonnes(bdd_titres, i, colonnes)

    # Rajout manuel d'une colonne intéressante
    #colonnes.append("nutrition_grade_fr")

    # On charge le dataset sur les colonnes qui nous ont intéressés dans la
    # fonction du dessus
    data = pd.read_csv(_FICHIER,
                       usecols=colonnes,
                       error_bad_lines=False,
                       engine='python',
                       sep=r'\t')

    # On supprime les lignes qui sont vides et n'ont que des "nan"
    # axis : {0 or ‘index’, 1 or ‘columns’},
    data = data.dropna(axis=1, how='all')
    data = data.dropna(how='all')

    # Je garde la description de bdd_data pour information
    #description = data.describe(include='all')

    # Suppression des colonnes qui ne remplissent pas les conditions posées
    for i in data:
        supprimer_colonnes(data, i)

    # Trouver le numéro de colonne qui nous sert d'ordonné dans l'affichage
    position_ordonne = data.columns.get_loc(_ORDONNEE)

    # Affichage des nuages de points avant traitement
    for i in data.columns.values[0:position_ordonne]:
        # Log
        print("Avant traitement")
        affichage_plot(data, i)

    # Log
    print("Fct traitement_data : \n")

    for nom_colonne in data.columns.values[0:position_ordonne]:
        # On garde les valeurs positives
        data = data[data[nom_colonne] >= 0]
        # On prends 98% de toutes les valeurs pour couper les grandes valeurs
        # farfelues
        data = data[data[nom_colonne] <= data[nom_colonne].quantile(0.98)]
        # Replace missing values with the mean if necessary
        #data = data.fillna(0)

    # Affichage des nuages de points après traitement
    for i in data.columns.values[0:position_ordonne]:
        # Log
        print("Après traitement")
        affichage_plot(data, i)

    print("\n\nLa base de données est nettoyée.\n\n")

    # création de l'objet pca
    #pca = decomposition.PCA(n_components = 2)
    # application de l'objet
    #pca.fit(data)
    #print(pca.explained_variance_ratio_.cumsum())
    #data2 = pca.transform(data)

    definir_importance(data)

if __name__ == "__main__":
    main()

    # test["nutrition_grade_fr"].value_counts(normalize = 1)

#============================================================================
# Tout ceci est inutile pour le moment
#     bdd_data["vitamin-b1_100g"].hist()
#     bdd_data.boxplot("vitamin-b1_100g")
#     bdd_data["vitamin-b1_100g"].value_counts()
#     pd.crosstab(bdd_data["vitamin-b1_100g"],bdd_data["nutrition-score-fr_100g"])
#     bdd_data = bdd_data[bdd_data["vitamin-b1_100g"]<10]
#     bdd_data.plot(kind = "scatter",x = "fat_100g",y = "nutrition-score-fr_100g")
#     bdd_data.plot(kind = "scatter",x = "energy_100g",y = "nutrition-score-fr_100g")
#     bdd_data.plot.hexbin(x = 'nutrition-score-fr_100g',y = 'fat_100g',gridsize = 100)
#     bdd_data['vitamin-d_100g'].quantile(.75)
#     from collections import Counter
#     a = data["nutrition_grade_fr"]
#     letter_counts = Counter(a)
#     df = pd.DataFrame.from_dict(letter_counts, orient = 'index')
#     df.plot(kind = 'bar')
#
#    colonnes = []
#    colonnes.append("nutrition_grade_fr")
#    colonnes.append("energy_100g")
#
#    test = pd.read_csv(FICHIER,
#                             usecols = colonnes,
#                             error_bad_lines = False,
#                             engine = 'python',
#                             sep = r'\t')
  ## Calculate the mean
    #mean = np.mean(data)

    ## Replace missing values with the mean
    #data = data.fillna(mean)
    # On garde les valeurs positives
    #data = data[data >= 0]


    #data = data[data < data.quantile(.98)]

    # Alternatively, you can also choose to interpolate missing values:
    # the interpolate() function will perform a linear interpolation
    # at the missing data points to “guess” the value that is most likely
    # to be filled in.
    #data = data.interpolate()

    # Replace missing values with the mean in case of remaining "NaN"
    #data = data.fillna(0)
    #data.groupby('nutrition-score-fr_100g').quantile([0.01,0.99])
    #result=df.groupby('V4')['V1','V2','V3'].apply(lambda x: x[(x> x.quantile(0.01)) & (x < x.quantile(0.99))]).dropna()
    #data=data.interpolate()
    #data=data.apply(lambda x: x[(x >= 0)])
    #data=data.apply(lambda x: x[(x < x.quantile(0.98))])


#============================================================================
  
