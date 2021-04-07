import os
import codecs
import pandas as pd
import codecs
import re
import pickle
from preprocessing import Preprocessing
import xml.etree.cElementTree as ET


cleaner=Preprocessing(lowercasing=True, number_removing=True, punctuation_removing=True, whitespaces_removing=True,
                      stopwords_removing=False, spell_checking=False, HTML_removing=True, URL_removing=True,
                      emoji_removing=False,stemming=False, lematizing = False, contraction_expanding=False)

def read_text(path):
	with codecs.open(path, 'r', encoding="utf-8") as f:
		return f.read()

def save_pkl(path, data):
    '''save pickle data into specified path '''
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
    '''load pickle data from specified path'''
    with open(path, "rb") as f:
        pkl = pickle.load(f)
    return pkl
    
def read_xml(path):
	content = open(path).read()
	tweets = []
	i = 0
	while True:
		i += 1
		start_documents = content.find('<document>')
		end_documents = content.find('</document>')
		tweets.append(' '.join(content[start_documents + 19 : end_documents-3].split()))
		content = content[end_documents+10:]
		if i == 100:
			break
	return tweets

def get_data(path):
    path_to_truth = os.path.join(path, 'truth.txt')
    truth = read_text(path_to_truth).split('\n')
    data = []
    for t in truth:
        file , label  = t.split(':::')
        tweets = read_xml(os.path.join(path, file+'.xml'))
        #for tweet in tweets:
        #    data.append([tweet, label.replace('\r','')])
        data.append([ '<TWEET>'.join(tweets), label.replace('\r','')])
        
    df = pd.DataFrame(data, columns=['tweet', 'label'])
    return df

def remove_specials(tweet):
    specials = ['RT', '#USER#', '#URL#', '#HASHTAG#']
    for special in specials:
        tweet=re.sub(r'{}'.format(special),"",tweet)
    return tweet

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def save_xml(path , user , l , t ):
    author = ET.Element("author" , id=str(user)[:-4], lang=l, type=t)
    tree = ET.ElementTree(author)
    tree.write(path)

def cleaning(tweets):
    lsts=['“', '”', '’']
    tweets = remove_specials(tweets)
    tweets = tweets.split('\n')
    tweets = [cleaner.clean(tweet, lsts=lsts) for tweet in tweets]
    return '. '.join(tweets)