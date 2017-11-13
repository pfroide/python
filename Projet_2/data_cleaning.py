"""
    Projet n°2.
    OpenFood
"""
#! /usr/bin/env python3
# coding: utf-8

# On importe les librairies dont on aura besoin pour ce tp
import pandas as pd

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

def fct_missing_data(data):
    
    # Compte les données manquantes par colonne
    missing_data = data.isnull().sum(axis=0).reset_index()
    
    # Change les noms des colonnes
    missing_data.columns = ['column_name', 'missing_count']
    
    # Crée une nouvelle colonne et fais le calcul en pourcentage des données
    # manquantes
    missing_data['filling_factor'] = (data.shape[0]-missing_data['missing_count']) / data.shape[0] * 100
    
    # Classe et affiche
    missing_data.sort_values('filling_factor').reset_index(drop = True)

    #
    print(missing_data)

def main():
    """
        Note : La première colonne et la dernière ont un " caché
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

    # Appel de le fonction qui va montrer les données manquantes
    fct_missing_data(data)
    
    # On supprime les lignes qui sont vides et n'ont que des "nan"
    # axis : {0 or ‘index’, 1 or ‘columns’},
    data = data.dropna(axis=1, how='all')
    data = data.dropna(how='all')

    # Suppression des colonnes qui ne remplissent pas les conditions posées
    for i in data:
        supprimer_colonnes(data, i)

    # export csv
    data.to_csv('C:\\Users\\Toni\\Desktop\\pas_synchro\\bdd_clean.csv')
