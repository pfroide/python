# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:16:04 2018

@author: Toni
"""

import datetime
import random as rd
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p5\\'
_FICHIERDATA = _DOSSIER + 'dataset_p5.csv'

def creation_client(data):
    """
    Fonction pour créer le client mystère
    """

    # 1 ligne = 1 client
    row_c = 0

    # Création d'un dataframe vide avec les bonnes colonnes
    df_cli = pd.DataFrame(columns=data.columns)
    del df_cli['labels']

    # Tirage au sort des valeurs de la facture
    df_cli.loc[row_c, 'nb_factures'] = 1
    df_cli.loc[row_c, 'somme_total'] = rd.randint(2, 500)
    df_cli.loc[row_c, 'nb_categorie_total'] = rd.randint(1, 500)
    df_cli.loc[row_c, 'nb_article_total'] = rd.randint(1, 100)
    df_cli.loc[row_c, 'day_of_week'] = rd.randint(0, 6)
    df_cli.loc[row_c, 'interval_jour_achat_n1'] = rd.randint(0, 12)
    df_cli.loc[row_c, 'interval_heure_achat_n1'] = rd.randint(0, 4)
    df_cli.loc[row_c, 'interval_moyenne_horaire'] = rd.randint(0, 4)
    df_cli.loc[row_c, 'valeur_facture_1'] = df_cli.loc[row_c, 'somme_total']

    # Déduction des autres
    df_cli.loc[row_c, 'mean_nb_article_facture'] = (
        df_cli.loc[row_c, 'nb_article_total']/df_cli.loc[row_c, 'nb_factures'])

    df_cli.loc[row_c, 'mean_somme_par_facture'] = (
        df_cli.loc[row_c, 'somme_total']/df_cli.loc[row_c, 'nb_factures'])

    df_cli.loc[row_c, 'mean_nb_categorie_facture'] = (
        df_cli.loc[row_c, 'nb_categorie_total']/df_cli.loc[row_c, 'nb_factures'])

    df_cli.loc[row_c, 'mean_somme_par_article'] = (
        df_cli.loc[row_c, 'somme_total']/df_cli.loc[row_c, 'nb_article_total'])

    df_cli.loc[row_c, 'ecart_moy_2_achats'] = 365
    df_cli.loc[row_c, 'ecart_min_2_achats'] = 365
    df_cli.loc[row_c, 'ecart_max_2_achats'] = 365

    # Tirage au sort de la date.
    startdate = datetime.date(2010, 12, 1)
    date = startdate + datetime.timedelta(rd.randint(1, 365))

    # Affichage de confirmation
    for i in df_cli:
        print(i, "\t", df_cli.loc[row_c, i])

    return df_cli

def main():
    """
    Fonction principale
    """

    # Lecture du dataset
    data = pd.read_csv(_FICHIERDATA, error_bad_lines=False)

    # Récupération de l'index
    data = data.set_index('Unnamed: 0')

    # Axe X
    data_x = data.copy()

    # On supprime les étiquettes de l'axe X
    del data_x['labels']

    # Axe Y = étiquettes
    data_y = data['labels']

    # Répartition Train/Test
    xtrain, xtest, ytrain, ytest = train_test_split(data_x, data_y, train_size=0.75)

    # Fit avec le classifieur choisi
    rfc = RandomForestClassifier().fit(xtrain, ytrain)

    # Déduction du label et affichage
    print(rfc.predict(creation_client(data))[0])
