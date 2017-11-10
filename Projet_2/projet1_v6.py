#! /usr/bin/env python3
# coding: utf-8

# On importe les librairies dont on aura besoin pour ce tp
import pandas as pd
import matplotlib.pyplot as plt

# Valeur limite des Nan acceptée
valeur_nan_limite=230000

# Référence en y du plot
ordonnee="nutrition-score-fr_100g"

# que faire des données manquantes ?
# affichage avant/après traitements
# faire un modèle de prédiction du score nutrionnel (continue/classe)

def affichage_plot(data,nom_colonne):
    
    global ordonnee
    
    #Log
    print("Affichage de la courbe")
    
    # Déliminations du visuel pour x
    xMax=max(data[nom_colonne])
    yMax=max(data[ordonnee])

    # Déliminations du visuel pour y
    xMin=min(data[nom_colonne])
    yMin=min(data[ordonnee])
    
    # création du nuage de point avec toujours la même ordonnée
    data.plot(kind="scatter",x=nom_colonne,y=ordonnee)
    
    # Affichage
    plt.grid(True)
    plt.xlim(xMin,xMax)
    plt.ylim(yMin,yMax)
    plt.legend()
    plt.show()
    
def affichage_points(data,nom_colonne):
    
    global ordonnee
    
    if nom_colonne.find('100g') > -1 and nom_colonne.find('nutrition') > -1 :
        
        # Log
        print("Fct affichage_points : Traitement de %s" % nom_colonne)
        
        print("Avant traitement")
        affichage_plot(data,nom_colonne)
        
        # On garde les valeurs positives
        data_temp=data[data[nom_colonne]>=0]
        
        # Suppresion des "Nan"
        data_temp[nom_colonne]=data_temp[nom_colonne].dropna()
        
        # On prends 98% de toutes les valeurs pour couper les grandes valeurs 
        # farfelues
        data_temp=data_temp[data_temp[nom_colonne]<data_temp[nom_colonne].quantile(.98)]
        
        print("Après traitement")
        affichage_plot(data_temp,nom_colonne)
    
def remplir_colonnes(data,nom_colonne,colonnes):
    # Log
    print("Fct remplir_colonnes : Traitement de : %s" % nom_colonne)
    
    # test du type de la colonne. IL n'y a que les valeurs numériques qui 
    # nous intéressent
    if data[nom_colonne].dtype=='float':
        # si "100g" est trouvé dans le nom de la colonne
        if nom_colonne.find('100g') != -1:
            colonnes.append(nom_colonne)
            print("Cette donnée est gardée")
        else:
            print("Cette donnée est exclue : pas de 100g")
        
    else:
        print("Cette donnée est exclue : pas un float")

def supprimer_colonnes(data,nom_colonne):
    # Log
    print("Fct supprimer_colonnes : Traitement de : %s" % nom_colonne)
    
    # nombre de valeurs "NaN"
    # .isnull().sum() = nombre par ligne
    # .isnull().sum().sum() = nombre total
    cpt_nan=data[nom_colonne].isnull().sum().sum()
    
    # S'il y a plus de valeur "Nan" que le chiffre défini, on vire la colonne
    if cpt_nan>(valeur_nan_limite):
        # Suprresion de la colonne
        del data[nom_colonne]
        
        # Log
        print("Cette donnée est exclue : elle contient %.0f 'NaN' " % cpt_nan)
    else:
        # Log
        print("Cette donnée est gardée : elle contient %.0f 'NaN' " % cpt_nan)
        
def histogramme(data,nom_colonne):
    
    from collections import Counter
    a = ['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'd', 'e', 'e', 'e', 'e', 'e']
    a = data["nutrition_grade_fr"]
    letter_counts = Counter(a)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    df.plot(kind='bar')
    
def main():
    # Note : La première colonne et la dernière ont un " caché
    # NB1 :
    # J'ai supprimé la dernière sans vérifier qu'elle était bonne
    # NB2 :
    # Si on additione tous les ingrédients, et qu'on est supérieur à 100g, 
    # il faudra sans doute supprimer la colonne
    
    # Lieu où se trouve le fichier
    Fichier='C:\\Users\\Toni\\Desktop\\pas_synchro\\bdd.csv'

    # Définition de la variable qui récupère le nom des colonnes
    colonnes=[]
    
    # On charge la première ligne du dataset
    bdd_titres = pd.read_csv(Fichier,
                             nrows=1,
                             error_bad_lines=False,
                             engine='python',
                             sep=r'\t')
    
    # Fonction qui va choisir les colonnes à récupérer suivant les crières 
    # définis
    for i in bdd_titres:
        remplir_colonnes(bdd_titres,i,colonnes)
    
    # Rajout manuel d'une colonne intéressante
    colonnes.append("nutrition_grade_fr")
    
    # On charge le dataset sur les colonnes qui nous ont intéressés dans la 
    # fonction du dessus
    data = pd.read_csv(Fichier,
                             usecols=colonnes,
                             error_bad_lines=False,
                             engine='python',
                             sep=r'\t')
    
    # On supprime les lignes qui sont vides et n'ont que des "nan"
    # axis : {0 or ‘index’, 1 or ‘columns’},
    data=data.dropna(axis=1,how='all')
    data=data.dropna(how='all')
    
    # Je garde la description de bdd_data pour information
    description = data.describe(include='all')

    # Suppression des colonnes vides
    for i in data:
        supprimer_colonnes(data,i)
        
    # Affichage des nuages de points
    for i in data:
        affichage_points(data,i)
        
    
    # test["nutrition_grade_fr"].value_counts(normalize=1)
    
# =============================================================================
# Tout ceci est inutile pour le moment        
#     bdd_data["vitamin-b1_100g"].hist()
#     bdd_data.boxplot("vitamin-b1_100g")
#     bdd_data["vitamin-b1_100g"].value_counts()
#     pd.crosstab(bdd_data["vitamin-b1_100g"],bdd_data["nutrition-score-fr_100g"])
#     bdd_data=bdd_data[bdd_data["vitamin-b1_100g"]<10]
#     bdd_data.plot(kind="scatter",x="fat_100g",y="nutrition-score-fr_100g")
#     bdd_data.plot(kind="scatter",x="energy_100g",y="nutrition-score-fr_100g")
#     bdd_data.plot.hexbin(x='nutrition-score-fr_100g',y='fat_100g',gridsize=100)
#     bdd_data['vitamin-d_100g'].quantile(.75)
#     from collections import Counter
#     a = data["nutrition_grade_fr"]
#     letter_counts = Counter(a)
#     df = pd.DataFrame.from_dict(letter_counts, orient='index')
#     df.plot(kind='bar')
#
#    colonnes=[]
#    colonnes.append("nutrition_grade_fr")
#    colonnes.append("energy_100g")
#    
#    test = pd.read_csv(Fichier,
#                             usecols=colonnes,
#                             error_bad_lines=False,
#                             engine='python',
#                             sep=r'\t')
# =============================================================================

    pass

if __name__ == "__main__":
    main()
