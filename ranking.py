import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from os import listdir
from btree_implementation_mk_IV import BTree, tree_insert, print_tree, store_tree, tree_search
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import math
import numpy as np
import time

path = './InvertedIndex/'
data_path = './Data/TelevisionNews/'
vector_path = './Vectors/'
files = listdir(path)

lem = WordNetLemmatizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


#Searh for and compute tf-idf scores of documents for free text queries.
def freetext_query(query, loaded_values):     
    
    #Load pickled values.
    tfidf = loaded_values['tfidf']
    doc_map = loaded_values['doc_map']
    vectorizer = loaded_values['vectorizer']
    vect_names = loaded_values['vect_names']

    #Result dictionary. Keys consist of unique document identifiers -> (FileName, RowNumber)
    doc_scores = dict()
   
    #Compute the query vector to get the term weights.
    query_vector = vectorizer.transform([' '.join(query)])
    query_vector = pd.DataFrame.sparse.from_spmatrix(query_vector)
    query_vector.columns = vect_names

    for filename in files:

        with open(path + filename, 'rb') as infile:
            B = pickle.load(infile)

            #Compute the scores for relevant documents term-wise.
            for term in query:
                
                #The term weight is the tf-idf weight of the term in the query vector.
                term_weight = query_vector[term][0]

                #Search for the term in the BTree.
                key = tree_search(B, term)
                if(key): #If the term exists in the corpus
                    for x in key[2]: #key[2] contains the posting list of the term. It is a list of tuples. 
                        
                        #Create the identifier for the document.
                        identifier = (filename+'.csv',x[0])

                        #Get the index corresonding to the document in the tf-idf matrix.
                        row = get_index(doc_map, identifier)

                        #Add to the score of the document if it is has already been encountered, otherwise create a new key and assign the score.
                        #The score is based on the term weight in the query vector and the tf-idf score of the term in the document vector.
                        if identifier in doc_scores:
                            doc_scores[identifier] += (term_weight * tfidf[term][row])
                        else:
                            doc_scores[identifier] = (term_weight * tfidf[term][row])

          

    return doc_scores


def phrase_query(query):

    #Result dictionary.
    term_frequency = dict()
    query_length = len(query)
    
    #Dictionary containing all documents that contain at least one of the terms in the phrase, keyed by unique identifier.
    docs = dict()

    start = time.perf_counter()
    for filename in files:
        with open(path + filename, 'rb') as infile:

            #B is the BTree for filename.
            B = pickle.load(infile)
            for term in query:

                #Search for the term in the BTree.
                key = tree_search(B, term)
                if(key): #If the term exists
                    #key[2] contains the posting list of the term. It is a list of tuples.
                    #Each tuple contains the row number of the documents and the list of positions in which the term is found.
                    for x in key[2]:
                        
                        #Create the unique identifier for the document -> (FileName, RowNumber)
                        identifier = (filename+'.csv',x[0])

                        #If the document has already been encountered, 
                        #increment the number of relevant terms it contains, and append the list of positions of the current term.
                        if identifier in docs:
                            docs[identifier]['positions'].append(x[1])
                            docs[identifier]['terms'] += 1
                        
                        #Otherwise, store the length of the document, the list of positions of the term, and the number of relevant terms.
                        else:
                            docs[identifier] = dict()
                            docs[identifier]['terms'] = 1
                            docs[identifier]['length'] = x[2]
                            docs[identifier]['positions'] = [x[1]]

    #Dictionary containing the frequencies of occurrence of the phrase in all relevant documents.
    doc_frequencies = dict()
                           
    start_out = time.perf_counter()    
    for identifier in docs.keys():

        #If the document contains all the terms in the phrase, check if the terms are in the correct order and positions.
        if docs[identifier]['terms'] == query_length:
            start = time.perf_counter()

            #Frequency of occurrence of the phrase in the document.
            frequency = 0

            #List of positions of the first phrase term.
            static = docs[identifier]['positions'][0]  

            for first in static:
                
                #Variable pointing to the positions of the terms after the first.
                second = 0
                
                #Indicates the term number in the phrase.
                array = 1

                while(array<query_length):

                    #If the array for the term is not empty
                    if len(docs[identifier]['positions'][array]) != 0:
                        #Store the first value in the position list of the term.
                        second = docs[identifier]['positions'][array][0]
                        notempty = 1

                        #If the term occurs before the first term, move the pointer to the next occurrence.
                        while(second < first and notempty == 1):
                            docs[identifier]['positions'][array].pop(0)
                            if len(docs[identifier]['positions'][array]) == 0:
                                notempty = 0
                            else:
                                second = docs[identifier]['positions'][array][0]
                    
                    #If the term is the right distance away from the first term, remove the position, and move to the next term array.
                    if (second - first) == array:
                        docs[identifier]['positions'][array].pop(0)
                        array+=1
                        if array == query_length:
                            frequency += 1
                    
                    #Otherwise move to the next position in the first term array.
                    else:
                        array = query_length
                    
            #If the phrase occurs in the document, store the frequency.
            if frequency != 0:
                doc_frequencies[identifier] = frequency

            end = time.perf_counter()
        
    end_out = time.perf_counter()

    if len(doc_frequencies)!=0:

        #Compute the score for each document based on the frequence of occurrence of the phrase, normalised by the document length.
        for identifier in doc_frequencies.keys():
            term_frequency[identifier] = (1+ math.log10(doc_frequencies[identifier]))/(docs[identifier]['length']+1-query_length)
                

    return term_frequency


#Find the index of the document in the tf-idf matrix using binary search.
def get_index(doc_map, identifier):
    
    l = 0
    r = len(doc_map)-1
    
    while(l<=r):
        mid = (l+r)//2
        if doc_map[mid] == identifier:
            return mid
        elif doc_map[mid] < identifier:
            l = mid+1
        else:
            r = mid-1
    
    return -1


#Rank the search results.
def rank(input, loaded_values, k = 10, isphrase = False):
    
    if isphrase == True:
        scores = phrase_query(input)
    else:
        scores = freetext_query(input, loaded_values)
    
    #Sort the results by score, and return the top k.
    df = pd.DataFrame.from_dict(scores, orient = "index",columns = ['Score'])
    df = df.sort_values('Score', ascending = False).head(k)
    time.perf_counter()

    return df


#Find the matching snippets in the CSV files based on the document identifiers.
def fetch_snippets(identifiers):
    snippets = []

    for identifier in identifiers:
        file_addr = data_path+identifier[0]
        row_no = int(identifier[1])
        csv_file = pd.read_csv(file_addr)
        snippet = csv_file['Snippet'][row_no]
        snippets.append((identifier,snippet))

    return snippets



