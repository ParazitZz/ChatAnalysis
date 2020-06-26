# coding: utf-8
import json
import bisect
import pickle
import cPickle
import numpy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

file='../../TelegramDataJson/result.json'
asupp = ["de", "le", "je", "pas", "c'est", "la", "les", "on", "un", "pour", "a", "en", "des", "et", "que", "du", "il", "est"]

def preprocess(NFentree, NFsortie) :
	fichier = open(NFentree, "r")
	fichier2 = open(NFsortie, "w")
	features_train = []
	labels_train = []
	allWords = []
	with open('../../TelegramDataJson/result.json') as json_file : 
		data = json.load(json_file)
		for c in data["chats"]["list"][0]["messages"]:
			if type(c["text"]) == list :
				pass
			else :
				try:
					for mot in c["text"].split() :
						bisect.insort(allWords, mot.replace("?",'').replace(".",'').replace('"','').replace('!','').replace(',','').lower())
				except KeyError as e:
					pass

	json_file.close()
	maxwords=wordsNumber(allWords)
	#print(maxwords)
	writeJSON('sortie.txt', maxwords)
	writeJSON('sortie.twt', recupMots(file, "Axel Meziani"))


def nombreToProba(dictio) : 
	comp=0
	for mot in dictio :
		comp = comp + dictio[mot]
	for mot in dictio : 
		nb = dictio[mot]
		dictio[mot] = float(nb)/float(comp)
	return(dictio)

def recupMots(filename, nom) :
	dictio = {}
	labels = recupLabels(filename)
	allWords=[]
	with open(filename) as json_file :
		data = json.load(json_file)
		for c in data["chats"]["list"][0]["messages"] : 
			try: 
				if type(c["text"]) == list :
					pass
				else :
					if c["from"] == nom :
						for mot in c["text"].split() :
							bisect.insort(allWords, mot.replace("?",'').replace(".",'').replace('"','').replace('!','').replace(',','').lower())
			except KeyError :
			  	pass
	json_file.close()
	numbers=wordsNumber(allWords)
	del numbers[""]
	printBests(numbers)
	writeJSON('sortie.tbt', nombreToProba(numbers))
	return(numbers)

def printBests(dictio) :
	'''for mot in asupp : 
		del dictio[mot]'''
	for i in range(20) :
		a=max(dictio)
		print("%s : %s" % (a[0], a[1]))
		del dictio[a[0]]

def recupLabels(filename) :
	noms=[]
	with open(filename) as json_file : 
		data = json.load(json_file)
		for c in data["chats"]["list"][0]["messages"]:
			try:
				if c["from"] not in noms : 
					noms.append(c["from"])
			except KeyError as e:
				pass
	return(noms)	

def suppDoubles(wordlist) :
	liste = []
	wd = "mmmmmmmmmmmmmmmmmmm"
	for word in wordlist :
		if wd == word :
			pass
		else : 
			liste.append(word)
			wd=word
	return(liste)

def wordsNumber(wordlist) :
	dictio = {}
	compt=0
	wd="mmmmmmmmmmmmmmmmmmmm"
	for word in wordlist :
		#print(compt , wd)
		if wd == word :
			compt=compt+1
		else :
			dictio[wd]=compt
			compt=1
			wd=word
	return(dictio)

def max(dictio) :
	a=0
	maxi="mmmmmmmm7"
	for mot in dictio :
		try:
			if dictio[mot] > a :
				maxi = mot
				a = dictio[mot]
		except TypeError as e:
			pass
	return([maxi, a])
			
def writeJSON(filename, data) :
	with open(filename, 'w') as json_file :
		json.dump(data, json_file)
	json_file.close()	

#preprocess("entree.txt","sortie.txt")
#preprocessing(file)
'''


dicto={"a" : 2,
	"b" : 3,
	"c" : 8,
	"d" : 2}

print(max(dicto))'''