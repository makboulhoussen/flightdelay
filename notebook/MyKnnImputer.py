# On importe toutes les librairies nécessaires
import os
import numpy as np
import pandas as pd
import math
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import r2_score
import glob
import pickle

# Classe permettant de remplacer les valeurs vides dans un dataframe
# Pour les features numériques, elle va calculer la meilleur valeur de k pour ensuite faire un remplacement
# basé sur les k plus proches voisins (distance euclidienne). Pour les variables catégorielles, elle va
# se baser sur les fréquences des variables pour remplacer les valeurs manquantes


class MyKnnImputer() :

	# init de la classe
	def __init__(self, data) :
		self.data = data
		self.data_num = self._getScaledNumericData(data)
		self.kvalues = {}
		self.loadBestKValuesForNumericFeatures()


	# retourne uniquement les colonnes de type numériques et après avoir centré et normé les données.
	def _getScaledNumericData(self, data) :
	    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	    data_num = data.select_dtypes(include=numerics)
	    data_num.fillna(data_num.mean())
	    data_num = (data_num - data_num.mean()) / (data_num.max() - data_num.min())
	    data_num.fillna(0, inplace=True)
	    return data_num


	# retourne la valeur moyenne des k plus proches voisin sur la feature col_name
	def _getNeighborsClosestValue(self, row_num, sample_data_num, sample_data, col_name,  k) :
		distances=euclidean_distances(sample_data_num, [row_num])
		closest = distances.argsort(axis=0)[:k].flatten()
		neighbors=[]
		for c in closest:
			neighbors.append(sample_data.iloc[c][col_name])
		value = np.mean(neighbors)
		return value

	############################################################
	#
	# Méthode pour mise à jour des valeurs manquantes
	#
	############################################################

	# met à jour les données manquantes dans le feature numérique col_name  en se basant sur les k plus proches voisins
	def updateNumericMissingValue(self, col_name, k):
		msk = self.data[col_name].isnull()
		target = self.data_num[msk]
		sample_data_num = self.data_num[~msk].sample(n=100)
		sample_data = self.data.ix[sample_data_num.index.values]
		for index, row in target.iterrows():
			v = self._getNeighborsClosestValue(row, sample_data_num, sample_data, col_name, k)
			self.data.at[index,col_name] = v


	# met à jour les données manquantes dans le feature catégorielle col_name en se basant sur les valeurs les plus fréquentes
	def updateCategoricalMissingValue(self, col_name):
	    frequentValue = self.data[col_name].value_counts().index[0]
	    msk = self.data[col_name].isnull()
	    target = self.data[msk]
	    for index, d in target.iterrows():
	        self.data.at[index,col_name] = frequentValue

	# met à jour toutes les données manquantes du dataframe.
	# Il se base sur les meilleurs valeurs de k pour les k plus proches voisins
	def updateMissingValues(self) :
		for column in self.data.select_dtypes(include=[np.number]).columns:
			if (self.data[column].isnull().values.any()):
				print("updating missing value in numerical : ", column)
				k = self.kvalues[column]
				self.updateNumericMissingValue(column,k)
		for column in self.data.select_dtypes(include=['object']).columns:
			if (self.data[column].isnull().values.any()):
				print("updating missing value in categorical : ", column)
				self.updateCategoricalMissingValue(column)


	######################################################################
	#
	# Méthode pour calculer la meilleur valeur de k par variable pour knn
	#
	######################################################################

	# retourne le score de précision dans le remplacement de valeur par les valeurs les plus proches
	def _getAccuracyScore(self, col_name, data_part, k) :
	    msk = data_part[col_name].isnull()
	    if (data_part[~msk].shape[0] <= 1) :
	    	return 0
	    data_sample = data_part[~msk].sample(n=100, replace=True)
	    x_test = self.data_num.ix[data_sample.index.values]
	    y_true = np.array(data_sample[col_name])
	    y_pred=[]
	    for index, t in x_test.iterrows():
	        exclude_msk = (data_sample.index == index)
	        data_train = data_sample[~exclude_msk]
	        data_train_num = x_test[~exclude_msk]
	        value = self._getNeighborsClosestValue(t, data_train_num, data_train,col_name, k)
	        y_pred.append(value)
	    y_pred=np.array(y_pred)
	    return (r2_score(y_true, y_pred))


	# retourne le score de précision moyen pour l'algorithme de remplacement des valeurs par les k plus proches
	# PArtitionnes les données en 10 folds et retourne le score moyen sur les 10 folds.    
	def _getCrossValidationAverageScore(self, col_name, k):
	    fold = 10
	    scores = []
	    partitions =  np.array_split(self.data, fold)
	    for d in partitions:
	        scores.append(self._getAccuracyScore(col_name, d, k))
	    return np.mean(scores)


	# Retourne la meilleure valeur de k pour la feature col_name pour les k plus proches voisins
	def getBestKValue(self, col_name) :
	    param_k = [3, 5, 7, 9]
	    score=-99
	    best_k = 3
	    for k in param_k :
	        s = self._getCrossValidationAverageScore(col_name, k)
	        if s > score :
	            score=s
	            best_k = k
	    return best_k


   # Retourne les meilleures valeurs de k pour les features numériques du dataframe
	def getBestKValuesForNumericVariables(self) :
		kvalues = {}
		for column in self.data.select_dtypes(include=[np.number]).columns:
			if (self.data[column].isnull().values.any()):
				print("processing ", column)
				k = self.getBestKValue(column)
				kvalues[column]=k
		return kvalues

	
	# Calcul les meilleures valeurs de k (knn) et sauvegarde le dictionnaire (col_name, k) dans le fichier 
	# kvalues.kpl
	def computeBestKValuesForNumericFeatures(self) :
		self.kvalues = self.getBestKValuesForNumericVariables()
		with open("kvalues.pkl", "wb") as myFile:
			pickle.dump(self.kvalues, myFile)

	
	# Charge le fichier dans lequels sont sauvegardées les meilleures valeurs de k (knn) pour chacune des colonnes.
	def loadBestKValuesForNumericFeatures(self) :
		with open('kvalues.pkl', 'rb') as f:
			self.kvalues = pickle.load(f)

	
	# retourne le dictionnaire qui donne la meilleure valeur de k (knn) pour chaque colonne numérique contenant
	# des valeurs vides.
	def getBestKValuesForNumericFeatures(self) :
		return self.kvalues


#path =r'.' # current dir
#datasets_path = glob.glob(path + "/2016_*.csv")
#flightsDB = pd.DataFrame()
#list_ = []
#for file_ in datasets_path:
#    df_ = pd.read_csv(file_,index_col=None, header=0,low_memory=False, error_bad_lines=False)
#    list_.append(df_)
#flightsDB = pd.concat(list_, ignore_index=True)
#df2 = flightsDB[[column for column in flightsDB if flightsDB[column].count() / len(flightsDB) >= 0.8]]
#print("List of dropped columns:", end=" ")
#for c in flightsDB.columns:
#    if c not in df2.columns:
#        print(c, end=", ")
#print('\n')
#flightsDB = df2.copy()

# Les variables suivantes doivent être vues comme des numériques, on force la transformation
#flightsDB[['YEAR','QUARTER','DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', \
#           'ORIGIN_CITY_MARKET_ID', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID','DEST_CITY_MARKET_ID', \
#           'DEP_TIME','TAXI_OUT', 'ARR_TIME']] = flightsDB[['YEAR','QUARTER','DAY_OF_MONTH', 'DAY_OF_WEEK',\
#                                                            'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID',\
#                                                            'ORIGIN_CITY_MARKET_ID', 'DEST_AIRPORT_ID',\
#                                                            'DEST_AIRPORT_SEQ_ID','DEST_CITY_MARKET_ID', \
#                                                            'DEP_TIME','TAXI_OUT', 'ARR_TIME']].apply(pd.to_numeric, errors='coerce')


#imputer = MyKnnImputer(flightsDB)
#imputer.updateMissingValues()
#print("null restant :")
#print(flightsDB.isnull().sum())
#print("done")




