# coding: utf-8
import json
import bisect
import numpy as np
import pandas as pd
from stop_words import get_stop_words

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

asupp = ["de", "le", "je", "pas", "c'est", "la", "les", "on", "un", "pour", "a", "en", "des", "et", "que", "du", "il", "est"]

tableLabels={}

def preprocessing(filename) :
	messages, authors, labels = get_data(filename)
	features_train, features_test, labels_train, labels_test = train_test_split(messages, authors, test_size=0.1, random_state=423)

	#print(listLabels)

	i=0
	for nom in labels :
		tableLabels[nom] = i
		i=i+1

	print(tableLabels)
	i=0
	for i in range(len(labels_train)):
		labels_train[i] = tableLabels[labels_train[i]]
	
	#print(labels_test)
	i=0
	for i in range(len(labels_test)):
		labels_test[i] = tableLabels[labels_test[i]]


	vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, stop_words = get_stop_words('french'))
	features_train_transformed = vectorizer.fit_transform(features_train)
	features_test_transformed  = vectorizer.transform(features_test)
	#writeJSON('sortie', vectorizer.get_feature_names())
	'''
	cv=CountVectorizer()
	# this steps generates word counts for the words in your docs
	word_count_vector=cv.fit_transform(features_train)

	tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
	tfidf_transformer.fit(word_count_vector)

	df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
	df_idf.sort_values(by=['idf_weights'])
	df_idf.to_excel("output.xlsx")
	#print(features_train_transformed)
	'''
	selector = SelectPercentile(f_classif, percentile=100)
	selector.fit(features_train_transformed, labels_train)
	features_train_transformed = selector.transform(features_train_transformed).toarray()
	features_test_transformed  = selector.transform(features_test_transformed).toarray()
	
	return features_train_transformed, features_test_transformed, labels_train, labels_test, vectorizer, selector

def miniPreprocess(string) :
	vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, stop_words = get_stop_words('french'))
	features_test_transformed  = vectorizer.transform(features_test)

def normalizing(feature, label) : 
	listLabels = suppDoubles(recupLabels(filename))
	vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, stop_words = get_stop_words('french'))
	feature_transformed = vectorizer.transform(feature)
	selector = SelectPercentile(f_classif, percentile=100)
	selector.fit(feature_transformed)
	features_train_transformed = selector.transform(feature_transformed).toarray()

	return(features_train_transformed)

def table(listLabels):
	i=0
	for nom in listLabels :
		tableLabels[nom] = i
		i=i+1

def get_data(filename) : 
	messages = []
	authors = []
	i=0
	with open(filename) as json_file :
		data = json.load(json_file)
		for msg in data["messages"]:
			if msg["text"] != "":
				authors.append(msg["from"])
				messages.append(msg["text"])
	json_file.close()

	numberOfMessages = {}
	for i in range(len(authors)):
		try:
			numberOfMessages[authors[i]] = numberOfMessages[authors[i]] + 1
		except KeyError:
			numberOfMessages[authors[i]] = 0

	i = 0
	indices = []
	to_delete = []
	for key in numberOfMessages:
		if numberOfMessages[key] <= 200:
			to_delete.append(key)

	for i in range(len(authors)):
		if authors[i] in to_delete:
			indices.append(i)

	authors = np.delete(authors, indices)
	messages = np.delete(messages, indices)

	labels = np.unique(np.array(authors))

	if len(authors) != len(messages):
		raise Exception("Problem in preprocessing, messages and authors unpaired")
	
	return(messages, authors, labels)



def recupLabels(filename) :
	noms=[]
	with open(filename) as json_file : 
		data = json.load(json_file)
		for c in data["chats"]["list"][1]["messages"]:
			try:
				if c["from"] not in noms : 
					noms.append(c["from"])
			except KeyError as e:
				pass
	return(noms)	


def writeJSON(filename, data) :
	with open(filename, 'w') as json_file :
		json.dump(data, json_file)
	json_file.close()	


def writenotJSON(filename, liste) : 
	with open(filename, 'w') as foo :
		for x in data:
			foo.write(x)
	foo.close()	

