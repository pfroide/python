"""
                'capric-acid_100g',
                'lauric-acid_100g',
                'myristic-acid_100g',
                'palmitic-acid_100g',
                'stearic-acid_100g',
                'arachidic-acid_100g',
                'behenic-acid_100g',
                'lignoceric-acid_100g',
                'cerotic-acid_100g',
                'montanic-acid_100g',
                'melissic-acid_100g',
                'monounsaturated-fat_100g',
                'polyunsaturated-fat_100g',
                'omega-3-fat_100g',
                'alpha-linolenic-acid_100g',
                'eicosapentaenoic-acid_100g',
                'docosahexaenoic-acid_100g',
                'omega-6-fat_100g',
                'linoleic-acid_100g',
                'arachidonic-acid_100g',
                'gamma-linolenic-acid_100g',
                'dihomo-gamma-linolenic-acid_100g',
                'omega-9-fat_100g',
                'oleic-acid_100g',
                'elaidic-acid_100g',
                'gondoic-acid_100g',
                'mead-acid_100g',
                'erucic-acid_100g',
                'nervonic-acid_100g',
                'trans-fat_100g',
                'cholesterol_100g',
                'carbohydrates_100g',
                'sugars_100g',
                'sucrose_100g',
                'glucose_100g',
                'fructose_100g',
                'lactose_100g',
                'maltose_100g',
                'maltodextrins_100g',
                'starch_100g',
                'polyols_100g',
                'fiber_100g',
                'proteins_100g',
                'casein_100g',
                'serum-proteins_100g',
                'nucleotides_100g',
                'salt_100g',
                'sodium_100g',
                'alcohol_100g',
                'vitamin-a_100g',
                'beta-carotene_100g',
                'vitamin-d_100g',
                'vitamin-e_100g',
                'vitamin-k_100g',
                'vitamin-c_100g',
                'vitamin-b1_100g',
                'vitamin-b2_100g',
                'vitamin-pp_100g',
                'vitamin-b6_100g',
                'vitamin-b9_100g',
                'folates_100g',
                'vitamin-b12_100g',
                'biotin_100g',
                'pantothenic-acid_100g',
                'silica_100g',
                'bicarbonate_100g',
                'potassium_100g',
                'chloride_100g',
                'calcium_100g',
                'phosphorus_100g',
                'iron_100g',
                'magnesium_100g',
                'zinc_100g',
                'copper_100g',
                'manganese_100g',
                'fluoride_100g',
                'selenium_100g',
                'chromium_100g',
                'molybdenum_100g',
                'iodine_100g',
                'caffeine_100g',
                'taurine_100g',
                'ph_100g',
                'fruits-vegetables-nuts_100g',
                'collagen-meat-protein-ratio_100g',
                'cocoa_100g',
                'chlorophyl_100g',
                'carbon-footprint_100g',
                'nutrition-score-fr_100g',
"""

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
    # J'ai supprimé la dernière sans vérifier qu'elle était bonne
    colonnes=[  'product_name',
                'vitamin-a_100g',
                'beta-carotene_100g',
                'vitamin-d_100g',
                'vitamin-e_100g',
                'vitamin-k_100g',
                'vitamin-c_100g',
                'vitamin-b1_100g',
                'vitamin-b2_100g',
                'vitamin-pp_100g',
                'vitamin-b6_100g',
                'vitamin-b9_100g']

    # On charge le dataset
    bdd_data = pd.read_csv(Fichier,error_bad_lines=False,
                           nrows=10000,
                           skip_blank_lines=True,
                           engine='python',
                           #usecols=colonnes,
                           sep=r'\t')

    # On supprime les lignes qui sont vides
    # axis : {0 or ‘index’, 1 or ‘columns’},
    bdd_data=bdd_data.dropna(how = 'all',thresh=160)
    bdd_data=bdd_data.dropna(axis=1,how = 'all')
    bdd_data.describe()
    bdd_data["vitamin-b1_100g"].hist()
    bdd_data.boxplot("vitamin-b1_100g")
    bdd_data["vitamin-b1_100g"].value_counts()
    pd.crosstab(bdd_data["vitamin-b1_100g"],bdd_data["nutrition-score-fr_100g"])
    bdd_data=bdd_data[bdd_data["vitamin-b1_100g"]<10]
    nutrition_grade_fr

    print(bdd_data)

    data=bdd_data['nutrition-score-uk_100g']

    for indice, valeur in enumerate(bdd_data):
        print ("bdd_data %r" % valeur)
        print(indice)
        variance=np.var(data[indice])
        ecartType=np.std(data[indice])
    
        print('variance = ', round(variance,3))
        print('ecartType = ', round(ecartType,3))
        
    # print(data.head())
    
    variance=np.var(data)
    ecartType=np.std(data)
    
    print('variance = ', round(variance,3))
    print('ecartType = ', round(ecartType,3))

    # Affichage du nuage de points avec couleur différente à chaque tour
    plt.grid(True)
    plt.plot(data,'ro',markersize=1)
    plt.legend()
    plt.show()
    
    #énumération des colonnes
    #print(bdd_data.columns)

    # Exemples d'affichage
    #print(bdd_data_numpy[2])
    #print(bdd_data.to_string())

    pass

if __name__ == "__main__":
    main()
