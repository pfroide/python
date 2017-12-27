# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Lieu où se trouve le fichier
Fichier='C:\\Users\\Toni\\Desktop\\movie_metadata.csv'
DossierTravail='C:\Users\Toni\python\python\Projet_3'

def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):        
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue        
        for s in [s for s in liste_keywords if s in liste]: 
            if pd.notnull(s): keyword_count[s] += 1
    #______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count

def main():
    
    # On charge le dataset
    data = pd.read_csv(Fichier)
    
    print("Données manquantes")
    # nombre de valeurs "NaN"
    # .isnull().sum() = nombre par ligne
    # .isnull().sum().sum() = nombre total
# =============================================================================
#     for nom_colonne in data:
#         # (df.shape[0] is the number of rows)
#         # (df.shape[1] is the number of columns)
#         pcent = 100*(data[nom_colonne].isnull().sum()/data.shape[0])
#         pcent=round(pcent,2)
#         print("{} \t {} \t {} %".format(nom_colonne, data[nom_colonne].isnull().sum(), pcent))
# =============================================================================
    
    # Compte les données manquantes par colonne
    missing_data = data.isnull().sum(axis=0).reset_index()
    
    # Change les noms des colonnes
    missing_data.columns = ['column_name', 'missing_count']
    
    # Crée une nouvelle colonne et fais le calcul en pourcentage des données
    # manquantes
    missing_data['filling_factor'] = (data.shape[0]-missing_data['missing_count']) / data.shape[0] * 100
    
    # Classe et affiche
    missing_data.sort_values('filling_factor').reset_index(drop = True)
    
    # compter tous les gens différents
    genre_labels = set()
    for s in data['genres'].str.split('|').values:
        genre_labels = genre_labels.union(set(s))
    
    # compter le nombre d'occurence
    keyword_occurences, dum = count_word(data, 'genres', genre_labels)
    keyword_occurences[:5]
    
    set_keywords2 = set()
    for liste_keywords in data['plot_keywords'].str.split('|').values:
        if isinstance(liste_keywords, float): 
            continue  # only happen if liste_keywords = NaN
        set_keywords2 = set_keywords2.union(liste_keywords)
    #_________________________
    # remove null chain entry
    #set_keywords2.remove('')

    keyword_occurences2, dum = count_word(data, 'plot_keywords', set_keywords2)
    keyword_occurences2[:5]
    
    set_keywords3 = set()
    for liste_keywords in data['director_name'].str.split('|').values:
        if isinstance(liste_keywords, float): 
            continue  # only happen if liste_keywords = NaN
        set_keywords3 = set_keywords3.union(liste_keywords)
    
    keyword_occurences3, dum = count_word(data, 'director_name', set_keywords3)
    keyword_occurences3[:5]
    
    db_names = []
    
    db_names.extend(data['actor_1_name'])
    db_names.extend(data['actor_2_name'])
    db_names.extend(data['actor_3_name'])
    
    df = pd.DataFrame(db_names,columns = ['name'])
                      
    #compte = {}.fromkeys(set(db_names),0)
    #for valeur in db_names:
    #    compte[valeur] += 1
#
    #print(compte)
    
    set_keywords4 = set()
    for liste_keywords in df['name'].str.split('|').values:
        if isinstance(liste_keywords, float): 
            continue  # only happen if liste_keywords = NaN
        set_keywords4 = set_keywords4.union(liste_keywords)
        
    keyword_occurences4, dum = count_word(df, 'name', set_keywords4)
    keyword_occurences4[:10]
    
    data['title_year'].describe()
    data['imdb_score'].describe()
    data['gross'].describe()
    
    words = dict()
    trunc_occurences = keyword_occurences[0:50]
    for s in trunc_occurences:
        words[s[0]] = s[1]
    
    plt.figure(figsize=(15,10))
    y_axis = [i[1] for i in trunc_occurences]
    x_axis = [k for k,i in enumerate(trunc_occurences)]
    x_label = [i[0] for i in trunc_occurences]
    plt.xticks(rotation=90, fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.xticks(x_axis, x_label)
    plt.ylabel("Nb. of occurences", fontsize = 18, labelpad = 10)
    plt.bar(x_axis, y_axis, align = 'center', color='b')
    plt.savefig('C:\\Users\\Toni\\python\\python\\Projet_3\\test2.png', dpi=100)

    #_______________________
    #plt.title("Keywords popularity",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 25)
    plt.show()
    
    
    #data['decade'] = data['title_year'].apply(lambda x:((x-1900)//10)*10)
    data['decade'] = data['title_year'].apply(lambda x:((x)//10)*10)
    
    #______________________________________________________________
    # Creation of a dataframe with statitical infos on each decade:
    test = data['title_year'].groupby(data['decade']).apply(get_stats).unstack()

    plt.pie(data['decade'])
#__________________________________________________________________
    # function that extract statistical parameters from a grouby objet:
def get_stats(gr):
    return {'min':gr.min(),'max':gr.max(),'count': gr.count(),'mean':gr.mean()}