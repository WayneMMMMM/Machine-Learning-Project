# Required Python Machine learning Packages
import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score


#load the dataset
dataset = pd.read_csv('student-mat.csv', header = 0, delimiter=' *; *', engine='python')
datasetPortion = dataset

#data reduction
toDrop = ['school','sex','age','address','famsize', 'Mjob','Fjob','Medu','Fedu',
	'reason','guardian','traveltime','studytime','schoolsup','paid','activities','nursery','higher',
	'internet','freetime','Dalc','Walc', 'health','G1','G2']
datasetPortion.drop(toDrop, inplace=True,axis=1)
datasetPortion = datasetPortion[['Pstatus','famsup','romantic','famrel','goout','failures','absences','G3']]
print ('\ndataset summuray after feature selection: \r')
print (datasetPortion.describe(include= 'all'))

#Encode string attributes into int
datasetPortion = datasetPortion.copy()
le = preprocessing.LabelEncoder()
pstatusCat = le.fit_transform(datasetPortion.Pstatus)
famsupCat = le.fit_transform(datasetPortion.famsup)
romanticCat = le.fit_transform(datasetPortion.romantic)
datasetPortion.loc[:,'pstatusCat'] = pstatusCat
datasetPortion.loc[:,'famsupCat'] = famsupCat
datasetPortion.loc[:,'romanticCat'] = romanticCat
dummyFields = ['Pstatus','famsup','romantic']
datasetPortion.drop(dummyFields,inplace=True,axis = 1)
datasetPortion = datasetPortion[['pstatusCat','famsupCat','romanticCat','famrel','goout','failures','absences','G3']]

#Put ordinal data into categories

bins = np.array([0,14])
datasetPortion.loc[:,'G3'] = np.digitize(datasetPortion.values[:,7],bins)
bins2 = np.array([0,4])
datasetPortion.loc[:,'absences'] = np.digitize(datasetPortion.values[:,6],bins2)
bins3 = np.array([0,4])
datasetPortion.loc[:,'famrel'] = np.digitize(datasetPortion.values[:,3],bins3)
bins4 = np.array([0,3])
datasetPortion.loc[:,'goout'] = np.digitize(datasetPortion.values[:,4],bins4)


#Optional scale dataset
#scaledFeatures = {}
#for each in ['pstatusCat','famsupCat','romanticCat','famrel','goout','failures','absences','G3']:
#	mean, std = datasetPortion[each].mean(), datasetPortion[each].std()
#	scaledFeatures[each] = [mean, std]
#	datasetPortion.loc[:,each] = (datasetPortion[each] - mean)/std 
#print ('\nscaled dataset: \r')
#print (datasetPortion)

#slice dataset into target and features, trainning set and test set
features = datasetPortion.values[:,:5]
target = datasetPortion.values[:,7]
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.2)

features_train = features_train.astype(int) 
target_train = target_train.astype(int)
target_test = target_test.astype(int)

#build model
clf = GaussianNB()
clf.fit(features_train, target_train)
#test model
traget_pred = clf.predict(features_test)
print ('\nthe accuracy is', accuracy_score(target_test,traget_pred, normalize = True))

#feature explanation 
print ('\nIf a student whose parents are seperate, who has no family support, involved in a romantic relationship, has poorest family relationship, and never hangs out with friends, then he',clf.predict_proba([[0,0,1,1,1]]))
print ('\nIf a student whose parents live together, who has family support, is not involved in a romantic relationship, has best family relationship, and always hangs out with friends, then he',clf.predict_proba([[1,1,0,2,2]]))

print ('\nattribute meanings:')
print ('Pstatus - parents cohabitation status (binary: "T" - living together or "A" - apart)')
print ('failures - number of past class failures (numeric: n if 1<=n<3, else 4)')
print ('famsup - family educational support (binary: yes or no)')
print ('romantic - with a romantic relationship (binary: yes or no)')
print ('famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)')
print ('goout - going out with friends (numeric: from 1 - very low to 5 - very high)')
print ('absences - number of school absences (numeric: from 0 to 93)')
print ('G3 - final grade (numeric: from 0 to 20, output target)')