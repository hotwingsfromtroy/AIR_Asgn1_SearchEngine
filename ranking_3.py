import pickle
import pandas as pd
import nltk
#nltk.download()
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

def get_tfidf_matrix(dataset):
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    vectorized_docs = vectorizer.fit_transform(dataset)
    print(vectorized_docs)
    names = vectorizer.get_feature_names()
    tfidf_matrix = pd.DataFrame.sparse.from_spmatrix(vectorized_docs).T
    tfidf_matrix.index = names
    return tfidf_matrix, vectorizer

def phrase_query(query, loaded_values):

    # tfidf = loaded_values['tfidf']
    doc_map = loaded_values['doc_map']

    term_frequency = dict()
    query_length = len(query)
    docs = {}

    start = time.perf_counter()
    for i in files:
        filename = i
        

        with open(path + filename, 'rb') as infile:
            B = pickle.load(infile)
            common = []
            for term in query:
                k = 0
                docIDs = set()
                key = tree_search(B, term)
                if(key):
                    for x in key[2]:
                        identifier = (filename+'.csv',x[0])
                        if identifier in docs:
                            docs[identifier]['positions'].append(x[1])
                            docs[identifier]['terms'] += 1
                        else:
                            docs[identifier] = dict()
                            docs[identifier]['terms'] = 1
                            docs[identifier]['length'] = x[2]
                            docs[identifier]['positions'] = [x[1]]

    end = time.perf_counter()
    print("Time taken to find all matching docs: ", end-start)
    print("Number of docs: ",len(docs))
    doc_frequencies = dict()
                           
    start_out = time.perf_counter()    
    for identifier in docs.keys():
        if docs[identifier]['terms'] == query_length:
            print("########### NEW DOC #############")
            start = time.perf_counter()
            frequency = 0
            second = 0
            static = docs[identifier]['positions'][0]                              
            for first in static:
                print("first:", first)
                array = 1
                while(array<query_length):
                    if len(docs[identifier]['positions'][array]) != 0:
                        second = docs[identifier]['positions'][array][0]
                        print("second:",second)
                        notempty = 1
                        while(second < first and notempty == 1):
                            print("second less than first")
                            docs[identifier]['positions'][array].pop(0)
                            if len(docs[identifier]['positions'][array]) == 0:
                                notempty = 0
                            else:
                                second = docs[identifier]['positions'][array][0]
                    
                    if (second - first) == array:
                        print("match")
                        docs[identifier]['positions'][array].pop(0)
                        array+=1
                        if array == query_length:
                            print("complete match")
                            frequency += 1
                    else:
                        array = query_length
                    
            if frequency != 0:
                doc_frequencies[identifier] = frequency

            end = time.perf_counter()
            print("Time taken to find the frequency for one doc: ",end-start)
        
    end_out = time.perf_counter()
    print("Total time taken to find the frequencies of phrases in documents: ", end_out - start_out)

    if len(doc_frequencies)!=0:
        for identifier in doc_frequencies.keys():
            term_frequency[identifier] = (1+ math.log10(doc_frequencies[identifier]))/(docs[identifier]['length']+1-query_length)
                

    return term_frequency


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



def freetext_query(query, loaded_values, top = 10):        # free text query
    # print(query)
    
    tfidf = loaded_values['tfidf']
    doc_map = loaded_values['doc_map']

    doc_scores = dict()
    # query_dict = dict()
    # for term in query:
    #     query_dict[term] = idf.loc[term,0]

    for i in files:
        filename = i 
        #docs = set()
        with open(path + filename, 'rb') as infile:
            B = pickle.load(infile)

            for term in query:
                key = tree_search(B, term)
                if(key):
                    for x in key[2]:
                        identifier = (filename+'.csv',x[0])
                        row = get_index(doc_map, identifier)
                        if identifier in doc_scores:
                            doc_scores[identifier] += tfidf[term][row]
                        else:
                            doc_scores[identifier] = tfidf[term][row]

          

    return doc_scores
  




def rank(input, loaded_values, top = 10, isphrase = False):
    if isphrase == True:
        scores = phrase_query(input, loaded_values)
    else:
        scores = freetext_query(input, loaded_values)
    
    start = time.perf_counter()
    df = pd.DataFrame.from_dict(scores, orient = "index",columns = ['Score'])
    end = time.perf_counter()
   # print("Time to convert results to df: ", end-start)
    #df = df['Score'].apply(lambda x: math.sqrt(x))
    start = time.perf_counter()
    df = df.sort_values('Score', ascending = False).head(top)
    time.perf_counter()
   # print("Time to sort the results and select top k: ", end-start)
    return df



def fetch_snippets(identifiers):
    snippets = []
    for identifier in identifiers:
        #print(identifier)
        file_addr = data_path+identifier[0]
        row_no = int(identifier[1])
        csv_file = pd.read_csv(file_addr)
        snippet = csv_file['Snippet'][row_no]
        snippets.append((identifier,snippet))
    return snippets



# d = [   
#         "this is some trial text",
#         "i am trying to figure out how to use some tfidf modules in python",
#         "it seems like it would be easier to write that code myself",
#         "i am not entirely sure what sort of sentence to give to accurately judge how it works",
#         "guess i will just adjust the sentence based on my requirements"
#     ]

# matrix, vect = get_tfidf_matrix(d)

# query_string = ["this is the trial sentence"]

# qv= vect.transform(query_string)

# # print("Matrix")
# # print(matrix)
# # print("Query Vect")
# # print(qv)
# #print(np.shape(qv))

# string = query_string[0].split(" ")

# alldocs = []
# for s in string:
#     for trialdoc in d:
#         if s in trialdoc:
#             alldocs.append(d.index(trialdoc))

# alldocs = set(alldocs)

# for a in alldocs:
#     dv = matrix[a].to_frame().T
#     score = cosine_similarity(qv,dv)
#     print(a, score)
#     print("#########################")

# l = [
#         [1,2,3,4,5],
#         [6,7,8,9,10],
#         [11,12,13,14,15]
#     ]
# df = pd.DataFrame(l)
# df.index = ['0','1','2']
# df.columns = ['a','b','c','d','e']
# #print(df)
# row = [df.loc['0']]
# print(np.shape(row))

# df = pd.DataFrame()
# df.columns = ['t','f']
# print(df)

# stop_words = set(stopwords.words('english'))

# q = "france, germany"
# q = [lem.lemmatize(x.lower()) for x in word_tokenize(q) if x.isalnum() and x not in stop_words]
# print(q)

