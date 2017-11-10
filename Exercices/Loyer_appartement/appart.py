# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pour la séparation Training/Testing
from sklearn.model_selection import train_test_split

# Régression linéaire
from sklearn import linear_model

# 3d
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

# On charge le dataset
house_data_raw = pd.read_csv('C:\\Users\\Toni\\Desktop\\WinPython-64bit-3.6.2.0Qt5\\notebooks\\data\\house_data.csv')
house_data_raw2 = house_data_raw[house_data_raw['price'] <7000]
house_data = house_data_raw2[house_data_raw['surface'] > 0]

#print (house_data)

# Séparation Train/Testing 20%/80%
xtrain, xtest, ytrain, ytest, ztrain, ztest = train_test_split(house_data['surface'], house_data['price'], house_data['arrondissement'], train_size=0.8)

couleur=['ro','go','bo','co','ko']
i=0

# On affiche le nuage de points dont on dispose
# On utilise set(ztrain) pour avoir une seule fois chaque valeur
for value in set(ztrain):
    # On fait 5 figures différentes
    #plt.figure(i+1)
    plt.subplot(1,5,i+1)
    # [ztrain==value] => condition pour prendre en compte
    xtrain_affiche=xtrain[ztrain==value]
    ytrain_affiche=ytrain[ztrain==value]
    # Pour que cela soit beau
    plt.grid(True)
    plt.title("Arr : " + str(value))
    plt.xlabel("Surface")
    plt.ylabel("Loyer")
    plt.xlim(0,150)
    plt.ylim(0,7000)
    # Affichage
    plt.plot(xtrain_affiche,ytrain_affiche,couleur[i], markersize=3)
    # Calcul du théta
    x = np.matrix([np.ones(xtrain.shape[0]), xtrain.as_matrix()]).T[ztrain==value]
    y = np.matrix(ytrain).T[ztrain==value]
    theta=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    print(theta)
    # Trucs marrants
    print("Loyer d'un 100m2 dans le " + str(value) + " arrondissement : " + str(round(theta.item(0) + theta.item(1) * 100,2)) + " euros")
    plt.plot([0,150], [theta.item(0),theta.item(0) + 150 * theta.item(1)],couleur[i],linestyle='--',label=round(theta.item(1),2))
    plt.legend()
    i=i+1

plt.show()
