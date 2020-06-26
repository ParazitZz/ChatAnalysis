# coding: utf-8
import json
import bisect
import numpy as np
import pandas as pd
import re, string
import random
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize


from usingClf import get_data


def mostMessages(messages, authors):
	numberOfMessages = {}
	for i in range(len(authors)):
		try:
			numberOfMessages[authors[i]] = numberOfMessages[authors[i]] + 1
		except KeyError:
			numberOfMessages[authors[i]] = 0
	
	m = 0
	for key in numberOfMessages:
		if numberOfMessages[key] > m:
			nom = key
			number = numberOfMessages[key]
			m = number

	return(nom, number)

def mostMdr(messages, authors):
	numberOfMdr = {}
	for i in range(len(messages)):
		if type(messages[i]) == str:
			regex = re.findall('mdr*', messages[i].lower())
			if regex != []:
				try:
					numberOfMdr[authors[i]] = numberOfMdr[authors[i]] + 1
				except KeyError:
					numberOfMdr[authors[i]] = 0

	m = 0
	for key in numberOfMdr:
		if numberOfMdr[key] > m:
			nom = key
			number = numberOfMdr[key]
			m = number

	return(nom, number)

def mostPositive(messages, authors):
	positive_tweets = twitter_samples.strings('positive_tweets.json')
	negative_tweets = twitter_samples.strings('negative_tweets.json')

	tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

	print(tweet_tokens[0])
	print(lemmatize_sentence(tweet_tokens[0]))
	stop_words = stopwords.words('english')

	positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
	negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

	positive_cleaned_tokens_list = []
	negative_cleaned_tokens_list = []

	for tokens in positive_tweet_tokens:
	    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

	for tokens in negative_tweet_tokens:
	    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

	all_pos_words = get_all_words(positive_cleaned_tokens_list)

	freq_dist_pos = FreqDist(all_pos_words)

	positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
	negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
	
	print(freq_dist_pos.most_common(10))
	
	positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

	negative_dataset = [(tweet_dict, "Negative")
	                     for tweet_dict in negative_tokens_for_model]

	dataset = positive_dataset + negative_dataset

	random.shuffle(dataset)

	train_data = dataset[:7000]
	test_data = dataset[7000:]

	classifier = NaiveBayesClassifier.train(train_data)

	print("Accuracy is:", classify.accuracy(classifier, test_data))

	print(classifier.show_most_informative_features(10))
	custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."

	custom_tokens = remove_noise(word_tokenize(custom_tweet))

	print(classifier.classify(dict([token, True] for token in custom_tokens)))

	custom_tweet = 'Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies'
	custom_tweet = 'Thank you for sending my baggage to CityX and flying me to CityY at the same time. Brilliant service. #thanksGenericAirline'

	return("Arthur Viens")

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":
	messages, authors, labels = get_data('result.json')
	print(f"The {len(labels)} labels are : {labels}")
	print(f"There are {len(messages)} messages to train on.")

	nom, number = mostMessages(messages, authors)
	print(f"Most messages sent on Wild : {nom, number}")
	nom, number = mostMdr(messages, authors)
	print(f"Most 'mdr' sen on Wild : {nom, number}")
	nom = mostPositive(messages, authors)
	print(f"Most positive {nom}")



	exit(-1)
	print(f"{nom} has sent the most messages : {messages} sent.")