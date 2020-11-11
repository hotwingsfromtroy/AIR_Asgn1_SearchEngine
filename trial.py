import pickle
from btree_implementation import print_tree, tree_search
from inverted_index import one_word_query, multi_word_query, phase_query
from os import listdir
import pickle
import pandas as pd
import nltk
# nltk.download()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from os import listdir
from btree_implementation import BTree, tree_insert, print_tree, store_tree
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
path = './InvertedIndex/'

lem = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

files = listdir(path)

data_path = './Data/TelevisionNews/'


query_1 = input('One word query: ')
one_word_query(lem.lemmatize(query_1))water


query_2 = input('Multi word query: ')
temp_query_1 = [ lem.lemmatize(x) for x in word_tokenize(query_2) if x.isalnum() and x not in stop_words ]
temp_query_2 = [i.lower() for i in temp_query_1]

multi_word_query(temp_query_2)


query_3 = input('Enter phrase query: ')

temp_query_1 = [ lem.lemmatize(x) for x in word_tokenize(query_3) if x.isalnum() and x not in stop_words ]
temp_query_2 = [i.lower() for i in temp_query_1]

print(temp_query_2)
phrase_query(temp_query_2)
