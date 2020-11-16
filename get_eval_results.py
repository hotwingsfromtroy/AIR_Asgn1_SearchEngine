import pickle
#from inverted_index import construct_inverted_index
from ranking import rank, fetch_snippets
from os import listdir
import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from os import listdir
from btree_implementation_mk_IV import BTree, tree_insert, print_tree, store_tree
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import time
from word_error_correction import get_correction
import json
path = './InvertedIndex/'

lem = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

files = listdir(path)

data_path = './Data/TelevisionNews/'

freetext = []
phrase = []

with open('queries.json','r') as json_file:
    queries_data = json.load(json_file)
    freetext = queries_data['freetext']
    phrase = queries_data['phrase']


vocab = ''
with open('tfidf_matrix', 'rb') as infile:
    matrix = pickle.load(infile)
#         # print(matrix.columns.tolist())
    vocab_list = matrix.columns.tolist()

#Read all the pickled data and store it in a dictionary.
tfidf_m = ''
with open("tfidf_matrix", "rb") as matrix_file:
    tfidf_m = pickle.load(matrix_file)

doc_m = ''
with open("document_mapping", "rb") as doc_file:
    doc_m = pickle.load(doc_file)

vect = ''
with open("tfidf_vectorizer", "rb") as vect_file:
    vect = pickle.load(vect_file)

loaded_values = dict()
loaded_values['tfidf'] = tfidf_m
loaded_values['doc_map'] = doc_m
loaded_values['vectorizer'] = vect
loaded_values['vect_names'] = vect.get_feature_names()


#Find the list of all words in the corpus.
vocab_list = tfidf_m.columns.tolist()

totals_f = []
totals_p = []

ft_dict = dict()
pt_dict = dict()

#MAIN MODULE
for text in freetext:
    start = time.perf_counter()
    
    temp_query = [lem.lemmatize(x.lower()) for x in word_tokenize(text) if x.isalnum() and x not in stop_words]
    mod_query = []
    for term in temp_query:
        if term in vocab_list:
            mod_query.append(term)
        else:
            corrections = get_correction(term, vocab_list, 3)
            if corrections:
                mod_query.append(list(corrections)[0])
    
    result = rank(mod_query, loaded_values, k = 100, isphrase = False)
    ids = result.index.values.tolist()
    snippets = fetch_snippets(ids)
    ft_results = []
    for snippet in snippets:
        ft_results.append(snippet[1])
    
    ft_dict[text] = ft_results
    
    end = time.perf_counter()
    totals_f.append(end-start)

f_avg = sum(totals_f)/len(totals_f)
ft_dict['Avg Time'] = f_avg

for text in phrase:
    start = time.perf_counter()
    
    temp_query = [lem.lemmatize(x.lower()) for x in word_tokenize(text) if x.isalnum() and x not in stop_words]
    mod_query = []
    for term in temp_query:
        if term in vocab_list:
            mod_query.append(term)
        else:
            corrections = get_correction(term, vocab_list, 3)
            if corrections:
                mod_query.append(list(corrections)[0])
    
    result = rank(mod_query, loaded_values, k = 100, isphrase = True)
    ids = result.index.values.tolist()
    snippets = fetch_snippets(ids)
    pt_results = []
    for snippet in snippets:
        pt_results.append(snippet[1])
    
    pt_dict[text] = pt_results
    
    end = time.perf_counter()
    totals_p.append(end-start)

p_avg = sum(totals_p)/len(totals_p)
pt_dict['Avg Time'] = p_avg

master_dict = dict()
master_dict['freetext'] = ft_dict
master_dict['phrase'] = pt_dict

with open('results.json','w') as json_file:
    json.dump(master_dict, json_file, indent = 4)
        