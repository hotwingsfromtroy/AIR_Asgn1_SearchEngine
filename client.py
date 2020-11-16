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
path = './InvertedIndex/'

lem = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

files = listdir(path)

data_path = './Data/TelevisionNews/'


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



#MAIN MODULE
query = input("Enter Query (Enter 'P' for Phrase Queries. | Enter 'Exit' to exit.): ")
while(query!="exit"):
    
    start = time.perf_counter()

    if(query.lower()!="p"):
        isphrase = False
    else:
        query = input("Enter Phrase: ")
        isphrase = True
        
    #Pre-process query.
    temp_query_1 = [lem.lemmatize(x.lower()) for x in word_tokenize(query) if x.isalnum() and x not in stop_words]

    #???????
    mod_query = []
    for term in temp_query_1:
        if term in vocab_list:
            mod_query.append(term)
        else:
            corrections = get_correction(term, vocab_list, 3)
            if corrections:
                mod_query.append(list(corrections)[0])
    
    #??????????
    if not mod_query:
        print('Unable to find the correct term in')
        query = input("Enter query: ")
        continue

    #Print the corrected query.
    print()
    print('Fetching results for query ', end = '')
    if isphrase:
        print('phrase ', end = '')
    print('"'+ ' '.join(mod_query)+'"')
    
    #Find the ranked results.
    result = rank(mod_query, loaded_values, isphrase = isphrase)

    #Get the document identifiers for the results.
    ids = result.index.values.tolist()

    #Fetch the result snippets.
    snippets = fetch_snippets(ids)

    end = time.perf_counter()
    
    for snippet in snippets:
        print("Identifier: ",snippet[0])
        print(snippet[1])
        print()
    
    print("TOTAL TIME TAKEN TO RETRIEVE SEARCH RESULTS: ", end-start)
    print()
    print()
    query = input("Enter query: ")





