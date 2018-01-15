# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 20:47:03 2018

@author: Toni
"""

import numpy as np

# Pour pouvoir afficher des graphiques:
import matplotlib.pyplot as plt

# Création d'un numpy array, de taille n, avec la porte gagnante (de 1 à 3)
def bonne_porte(n):
    return np.random.randint(1, 4, size=n)

# Création d'un numpy array, de taille n, avec le choix du candidat (de 1 à 3)
def premier_choix(n):
    return np.random.randint(1, 4, size=n)

# Création d'un numpy array, qui représente la porte qui sera ouverte
def suppression_porte(guesses, prizedoors):
    return [np.setdiff1d([1, 2, 3], [prizedoors[i], guesses[i]])[0] for i in range(len(prizedoors))]

# Création d'un numpy array de la porte choisie en deuxième choix par le candidat
def deuxieme_choix(guesses, goatdoors):
    return [np.setdiff1d([1, 2, 3], [guesses[i], goatdoors[i]])[0] for i in range(len(guesses))]

N = 10000

# Sans changer de porte :
nb_bonne_porte = bonne_porte(N)
nb_premier_choix = premier_choix(N)
resultat_sans_changer = np.sum(nb_premier_choix==nb_bonne_porte)
pc_resultat_sans_changer = resultat_sans_changer/N*100

print("Sans changer de porte : %.2f" % pc_resultat_sans_changer)

# En changeant de porte :
nb_bonne_porte = bonne_porte(N)
nb_premier_choix = premier_choix(N)
nb_suppression = suppression_porte(nb_premier_choix, nb_bonne_porte)
nb_deuxieme_choix = deuxieme_choix(nb_premier_choix, nb_suppression)
resultat_en_changeant = np.sum(nb_deuxieme_choix==nb_bonne_porte)
pc_resultat_en_changeant = resultat_en_changeant*100/N
print("En changeant de porte : %.2f" % pc_resultat_en_changeant)

# plot renvoie un objet, que l'on pourra manipuler plus tard pour
# personnaliser le graphique

plot = plt.bar([1,2],
               [resultat_en_changeant,resultat_sans_changer], 
                tick_label=["Changer","Garder"])
    