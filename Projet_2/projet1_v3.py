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

def limites(bdd_data,nom_colonne):
    # test du type de la colonne
    if bdd_data[nom_colonne].dtype=='float':
    #if isinstance(bdd_data[nom_colonne],float):
        if nom_colonne.find('100g'):
            #bdd_data=bdd_data[bdd_data[nom_colonne]<=100][bdd_data[nom_colonne]>=0]
            print("C'est le bon type")
            #print(bdd_data[nom_colonne].head)
            #bdd_data_temp=bdd_data[nom_colonne].quantile(.75)
            bdd_data_temp=bdd_data[bdd_data[nom_colonne]<bdd_data[nom_colonne].quantile(.90)]
            
            X=bdd_data_temp[nom_colonne]
            y=bdd_data_temp["nutrition-score-fr_100g"]
            
            regr = bdd_data.LinearRegression()
            regr.fit(X, y)
            regr.predict(bdd_data_temp)

            # liste avec critères
            #print(bdd_data[nom_colonne].loc[bdd_data[nom_colonne]>=10][bdd_data[nom_colonne]<=25])
            bdd_data_temp.plot(kind="scatter",x=nom_colonne,y="nutrition-score-fr_100g")
            
            plt.show()
            
            """for i,j in enumerate(bdd_data[nom_colonne]):
                #if j=="NaN":
                #    print(i,j)
                #    bdd_data[j]=0
                print(j)
                if j>=100:
                    print('XXXXXXXXXXXXXXXXXXXXXX = ', bdd_data[nom_colonne][i:j])
                    #print(i,j)
                    #bdd_data[j]=100
                    #print(i,j)
            """
    else:
        print("Ce n'est pas bon type")
        
    # Si on additione tous les ingrédients, et qu'on est supérieur à 100g, c'est faux
    
def main():
    # Lieu où se trouve le fichier
    Fichier='C:\\Users\\Toni\\Desktop\\pas_synchro\\bdd.csv'
    
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
                'vitamin-b9_100g',
                'fat_100g',
                'nutrition-score-fr_100g']
    
    # On charge le dataset
    bdd_data = pd.read_csv(Fichier,
                           usecols=colonnes,
                           error_bad_lines=False,
                           engine='python',
                           sep=r'\t')

    # save base d'origine
    bdd_data_origin=bdd_data.copy()
    
    # On supprime les lignes qui sont vides
    # axis : {0 or ‘index’, 1 or ‘columns’},
    # Voir si celle-la est à faire : bdd_data=bdd_data.dropna(how = 'all',thresh=160)
    bdd_data=bdd_data.dropna(axis=1,how='all')
    
    # Je garde la description de bdd_data
    description = bdd_data.describe(include='all')
    
    bdd_data=np.array(bdd_data)
    
    # Je supprime ce qui est supérieur à 100g et inférieur à 0g
    # Ce n'est pas forcément une bonne idée
    bdd_data=bdd_data[bdd_data['fat_100g']<=100][bdd_data['fat_100g']>=0]
    #bdd_data=bdd_data[bdd_data['fat_100g']>=0]
    
    # appel fonction
    for i in bdd_data:
        limites(bdd_data,i)
    
    # maximums et minimuns
    maximums=bdd_data.max()
    minimums=bdd_data.min()
    
    energy_100g
    
    bdd_data["vitamin-b1_100g"].hist()
    bdd_data.boxplot("vitamin-b1_100g")
    bdd_data["vitamin-b1_100g"].value_counts()
    pd.crosstab(bdd_data["vitamin-b1_100g"],bdd_data["nutrition-score-fr_100g"])
    bdd_data=bdd_data[bdd_data["vitamin-b1_100g"]<10]
    bdd_data.plot(kind="scatter",x="fat_100g",y="nutrition-score-fr_100g")
    bdd_data.plot(kind="scatter",x="energy_100g",y="nutrition-score-fr_100g")
    bdd_data.plot.hexbin(x='nutrition-score-fr_100g',y='fat_100g',gridsize=100)
    bdd_data['vitamin-d_100g'].quantile(.75)
    
    print(bdd_data)

    data=bdd_data['nutrition-score-fr_100g']

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
