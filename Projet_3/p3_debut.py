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
DossierTravail='C:\\Users\\Toni\\python\\python\\Projet_3'

#__________________________________________________________________
    # function that extract statistical parameters from a grouby objet:
def get_stats(gr):
    return {'min':gr.min(),'max':gr.max(),'count': gr.count(),'mean':gr.mean()}

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

def afficher_plot(trunc_occurences):
    
    words = dict()
    
    for s in trunc_occurences:
        words[s[0]] = s[1]
    
    plt.figure(figsize=(15,10))
    y_axis = [i[1] for i in trunc_occurences]
    x_axis = [k for k,i in enumerate(trunc_occurences)]
    x_label = [i[0] for i in trunc_occurences]
    plt.xticks(rotation=90, fontsize = 10)
    plt.xticks(x_axis, x_label)
    
    plt.yticks(fontsize = 10)
    plt.ylabel("Nb. of occurences", fontsize = 18, labelpad = 10)
    
    plt.bar(x_axis, y_axis, align = 'center', color='b')
    
    plt.savefig('C:\\Users\\Toni\\python\\python\\Projet_3\\test2.png', dpi=100)
    
    plt.title("Keywords popularity", fontsize = 25)
    plt.show()
    
def main():
    
    # On charge le dataset
    data = pd.read_csv(Fichier)
    
    print("Données manquantes")
    # nombre de valeurs "NaN"
    # .isnull().sum() = nombre par ligne
    # .isnull().sum().sum() = nombre total    
    # Compte les données manquantes par colonne
    missing_data = data.isnull().sum(axis=0).reset_index()
    
    # Change les noms des colonnes
    missing_data.columns = ['column_name', 'missing_count']
    
    # Crée une nouvelle colonne et fais le calcul en pourcentage des données
    # manquantes
    missing_data['filling_factor'] = (data.shape[0]-missing_data['missing_count']) / data.shape[0] * 100
    
    # Classe et affiche
    missing_data.sort_values('filling_factor').reset_index(drop = True)
    
    # compter tous les genres différents
    genre_list = set()
    
    for s in data['genres'].str.split('|').values:
        genre_list = genre_list.union(set(s))
    
    # compter le nombre d'occurence de ces genres
    genre_list, dum = count_word(data, 'genres', genre_list)
    
    # compter les languages
    language_list = set()
    
    for word in data['language'].str.split('|').values:
        if isinstance(word, float): 
            continue  # only happen if liste_keywords = NaN
        language_list = language_list.union(word)
    
    # compter le nombre d'occurence des languages
    language_list, dum = count_word(data, 'language', language_list)
    
    # compter les languages
    country_list = set()
    
    for word in data['country'].str.split('|').values:
        if isinstance(word, float): 
            continue  # only happen if liste_keywords = NaN
        country_list = country_list.union(word)
    
    # compter le nombre d'occurence des languages
    country_list, dum = count_word(data, 'country', country_list)
    
    # compter les languages
    rating_list = set()
    
    for word in data['content_rating'].str.split('|').values:
        if isinstance(word, float): 
            continue  # only happen if liste_keywords = NaN
        rating_list = rating_list.union(word)
    
    # compter le nombre d'occurence des languages
    rating_list, dum = count_word(data, 'content_rating', rating_list)
    
    # compter tous les mot définissants les films différents
    keywords_list = set()
    
    for word in data['plot_keywords'].str.split('|').values:
        if isinstance(word, float): 
            continue  # only happen if liste_keywords = NaN
        keywords_list = keywords_list.union(word)
    
    # remove null chain entry
    #set_keywords2.remove('')

    # en compter le nombre d'occurence
    keywords_list, dum = count_word(data, 'plot_keywords', keywords_list)
    
    # compter tous les directeurs de films
    directors_list = set()
    
    for word in data['director_name'].str.split('|').values:
        if isinstance(word, float): 
            continue  # only happen if word = NaN
        directors_list = directors_list.union(word)
    
    # en compter le nombre d'occurence
    directors_list, dum = count_word(data, 'director_name', directors_list)
    
    db_names = []    
    db_names.extend(data['actor_1_name'])
    db_names.extend(data['actor_2_name'])
    db_names.extend(data['actor_3_name'])
    
    df = pd.DataFrame(db_names,columns = ['name'])

    # compter tous les acteurs de films
    actors_list = set()
    
    for word in df['name'].str.split('|').values:
        if isinstance(word, float): 
            continue  # only happen if liste_keywords = NaN
        actors_list = actors_list.union(word)
    
    # en compter le nombre d'occurence
    actors_list, dum = count_word(df, 'name', actors_list)

    afficher_plot(genre_list[0:50])
    afficher_plot(keywords_list[0:50])
    afficher_plot(directors_list[0:50])
    afficher_plot(actors_list[0:50])
    afficher_plot(language_list[0:50])
    afficher_plot(country_list[0:50])
    afficher_plot(rating_list[0:50])
    
    # Affichage des décades
    data['decade'] = data['title_year'].apply(lambda x:((x)//10)*10)
    
    # Creation of a dataframe with statitical infos on each decade:
    test = data['title_year'].groupby(data['decade']).apply(get_stats).unstack()
    
    sizes = test['count'].values / (data['title_year'].count()) * 100
    
    # pour le camembert
    explode = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    # affichage du camembert
    plt.pie(sizes, explode=explode, labeldistance=1.2, labels=round(test['min'],0), shadow=False, startangle=0, autopct = lambda x:'{:1.0f}%'.format(x) if x > 5 else '')
    
    # Liste des noms complets à analyser
    Alphabet = []    
    Alphabet.append('num_critic_for_reviews')    
    Alphabet.append('num_user_for_reviews')
    Alphabet.append('duration')
    Alphabet.append('gross')
    Alphabet.append('budget')
    Alphabet.append('imdb_score')
    Alphabet.append('movie_facebook_likes')
    Alphabet.append('cast_total_facebook_likes')
    Alphabet.append('num_voted_users')

    
     # Affichage des imdb_score par 10
    data['imdb_score10'] = data['imdb_score'].apply(lambda x:round(x,0))
    
    # Creation of a dataframe with statitical infos on each decade:
    test = data['imdb_score'].groupby(data['imdb_score10']).apply(get_stats).unstack()
    sizes = test['count'].values / (data['imdb_score'].count()) * 100
    
    # pour le camembert
    explode = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    # affichage du camembert
    plt.pie(sizes, explode=explode, labeldistance=1.2, labels=round(test['min'],0), shadow=False, startangle=0, autopct = lambda x:'{:1.0f}%'.format(x) if x > 5 else '')
