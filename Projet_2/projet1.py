#! /usr/bin/env python3
# coding: utf-8

# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Pour la séparation Training/Testing
from sklearn.model_selection import train_test_split

# Régression linéaire
from sklearn import linear_model

def main():
    # Lieu où se trouve le fichier
    Fichier='C:\\Users\\Toni\\Desktop\\bdd.csv'
    
    # Note : La première colonne et la dernière ont un " caché
    colonnes=['product_name','generic_name','trans-fat_100g','cholesterol_100g','carbohydrates_100g','sugars_100g','sucrose_100g','glucose_100g','fructose_100g','lactose_100g','maltose_100g']

    # On charge le dataset
    test = pd.read_csv(Fichier,error_bad_lines=False,
                           nrows=1,
                           header=None,
                           engine='python',
                           sep=r'\t')

    test_numpy=np.array(test)

    liste=[]
    """
    #link=re.compile(b'^.*100g$')
    link=re.compile(b'^.*$')
    print(test_numpy)
    
    #Add the b there, it makes it into a bytes object
    lettre=link.findall(test_numpy)
    print(lettre)
    """
    
    for lettre in test:
        mot=str(test[lettre])
        print(mot)
        if mot.find('100g') >0:
            liste.append(mot)
    print (liste)
    
    #re.findall(r'^.*100g$',test_numpy[lettre])
        
    """
    liste=[]
    liste.append()
    
    colonnes=re.findall(r'^.*100g$',test)
    print(colonnes)

    # On charge le dataset
    bdd_data = pd.read_csv(Fichier,error_bad_lines=False,
                           nrows=10000,
                           engine='python',
                           sep=r'\t',
                           usecols=colonnes)

    # On supprime les lignes qui sont vides
    #bdd_data=bdd_data.dropna(how = 'all')
    bdd_data=bdd_data.dropna(axis=1,how = 'all')

    print(bdd_data)
    
    #énumération des colonnes
    #print(bdd_data.columns)
    """
    
    """
    bdd_data_numpy=np.array(bdd_data)
    print(bdd_data_numpy)
    """

    # Exemples d'affichage
    #print(bdd_data_numpy[2])
    #print(bdd_data.to_string())

    pass

if __name__ == "__main__":
    main()
