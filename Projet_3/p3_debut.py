# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lieu où se trouve le fichier
Fichier='C:\\Users\\Toni\\Desktop\\movie_metadata.csv'

def main():
    
    # On charge le dataset
    data = pd.read_csv(Fichier)
    
    print("Données manquantes")
    # nombre de valeurs "NaN"
    # .isnull().sum() = nombre par ligne
    # .isnull().sum().sum() = nombre total
    for nom_colonne in data:
        # (df.shape[0] is the number of rows)
        # (df.shape[1] is the number of columns)
        pcent = 100*(data[nom_colonne].isnull().sum()/data.shape[0])
        pcent=round(pcent,2)
        print("{} \t {} \t {} %".format(nom_colonne, data[nom_colonne].isnull().sum(), pcent))
    
    # Compte les données manquantes par colonne
    missing_data = data.isnull().sum(axis=0).reset_index()
    
    # Change les noms des colonnes
    missing_data.columns = ['column_name', 'missing_count']
    
    # Crée une nouvelle colonne et fais le calcul en pourcentage des données
    # manquantes
    missing_data['filling_factor'] = (data.shape[0]-missing_data['missing_count']) / data.shape[0] * 100
    
    # Classe et affiche
    missing_data.sort_values('filling_factor').reset_index(drop = True)
