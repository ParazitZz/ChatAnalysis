#!/usr/bin/python

    
import sys
from time import time
from usingClf import preprocessing

file='result.json'
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels



from sklearn.metrics import accuracy_score
from usingClf import tableLabels
#########################################################
### your code goes here ###

def SVMClassifier(features_train, labels_train, features_test, labels_test, Cs=1, Gam='auto') :
	features_train = features_train[:len(features_train)/20]
	labels_train = labels_train[:len(labels_train)/20]
	from sklearn.svm import SVC
	clf = SVC(gamma=Gam, kernel='linear', C=Cs)
	t0 = time()
	clf.fit(features_train, labels_train)
	print(f"training time: {round(time()-t0, 3)} s")
	t1 = time()
	pred = clf.predict(features_test)
	print(f"prediction time : {round(time()-t1, 3)} s")
	print(f"Accuracy of SVM with C = {Cs}, {accuracy_score(labels_test, pred)}")
	return clf

def GNBClassifier(features_train, labels_train, features_test, labels_test) :
	from sklearn.naive_bayes import GaussianNB
	clf=GaussianNB()
	t0 = time()
	clf.fit(features_train, labels_train)
	print(f"training time: {round(time()-t0, 3)}s")
	t1 = time()
	pred=clf.predict(features_test)
	print(f"prediction time : {round(time()-t1, 3)}s")
	print(f"Accuracy of naive_bayes : {accuracy_score(pred, labels_test)}")
	return clf

def TreeClassifier(features_train, labels_train, features_test, labels_test) :
	features_train = features_train[:len(features_train)/10]
	labels_train = labels_train[:len(labels_train)/10]
	from sklearn import tree 
	clf = tree.DecisionTreeClassifier(min_samples_split=50)
	t0 = time()
	clf.fit(features_train, labels_train)
	print(f"training time: {round(time()-t0, 3)}s")
	t1 = time()
	pred=clf.predict(features_test)
	print(f"prediction time : {round(time()-t1, 3)}s")
	print(f"Accuracy of Decision Tree : {accuracy_score(pred, labels_test)}")
	return clf

def KNN(features_train, labels_train, features_test, labels_test) :
	from sklearn.neighbors import KNeighborsClassifier
	clf = KNeighborsClassifier()
	clf.fit(features_train, labels_train)
	t0 = time()
	#print "training time:", round(time()-t0, 3), "s"
	t1 = time()
	pred=clf.predict(features_test)
	#print "prediction time : ", round(time()-t1, 3), "s"
	print(f"Accuracy of KNN : {accuracy_score(pred, labels_test)}")
	return clf

def Adaboost(features_train, labels_train, features_test, labels_test) :
	from sklearn.ensemble import AdaBoostClassifier
	clf = AdaBoostClassifier()
	clf.fit(features_train, labels_train)
	t0 = time()
	#print "training time:", round(time()-t0, 3), "s"
	t1 = time()
	pred=clf.predict(features_test)
	#print "prediction time : ", round(time()-t1, 3), "s"
	print(f"Accuracy of Adaboost : {accuracy_score(pred, labels_test)}")
	return clf

def RandomForest(features_train, labels_train, features_test, labels_test) :
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(min_samples_split=3)
	clf.fit(features_train, labels_train)
	t0 = time()
	#print "training time:", round(time()-t0, 3), "s"
	t1 = time()
	pred=clf.predict(features_test)
	#print "prediction time : ", round(time()-t1, 3), "s"
	print(f"Accuracy of RandomForest : {accuracy_score(pred, labels_test)}")
	return clf

"""print labels_train"""
'''
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
from sklearn.metrics import accuracy_score
t1 = time()
pred=clf.predict(features_test)

print "prediction time : ", round(time()-t1, 3), "s"
print(accuracy_score(pred, labels_test))
'''

if __name__ == "__main__":
	features_train, features_test, labels_train, labels_test, vectorizer, selector= preprocessing(file)
	
	clf = GNBClassifier(features_train, labels_train, features_test, labels_test)
	clf = SVMClassifier(features_train, labels_train, features_test, labels_test, 15, 10)
	clf = TreeClassifier(features_train, labels_train, features_test, labels_test)
	clf = KNN(features_train, labels_train, features_test, labels_test)
	clf = Adaboost(features_train, labels_train, features_test, labels_test)
	clf = RandomForest(features_train, labels_train, features_test, labels_test)


	string = ""
	while string != "stop" :
		string = raw_input("Entrez une phrase.\n")
		print("La phrase est : ")
		print(f"   {string}")
		feat = vectorizer.transform([string])
		feat = selector.transform(feat).toarray()
		pred = clf.predict(feat)
		print("La personne la plus susceptible d'avoir ecrit ce message est : ")
		print(tableLabels.keys()[tableLabels.values().index(pred)])
		print("--------------------")
	#########################################################

