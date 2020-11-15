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

def phrase_query(query):

    term_frequency = dict()
    length = 10

    for i in files:
        filename = i
        docs = {}

        with open(path + filename, 'rb') as infile:
            B = pickle.load(infile)
            common = []
            for term in query:
                docIDs = set()
                key = tree_search(B, term)
                if(key):
                    # print(posting_list)
                    docs[term] = {}
                    no_of_items = key[1]
                    for i in range(no_of_items):
                        docIDs.add((key[2][i][0],key[2][i][2])) #length is a placeholder. this is where the document length needs to go. the rest of the code has been modified. 
                        docs[term][key[2][i][0]] = key[2][i][1]
                common.append(docIDs)

            inter = common[0]               # for performing intersection
            for j in common:
                inter = inter & j

            # inside inter, we get all row numbers in one file having all the terms in the query


            for row in inter:  # for each row
                print("0:",row[0])
                print("1:",row[1])
                identifier = (filename+'.csv',row[0])
                check = []
                for term in query:
                    check.append(docs[term][row[0]])

                temp = set(check[0])
                for k in range(1, len(check)):
                    for m in range(len(check[k])):
                        check[k][m] -= k
                    temp &= set(check[k])

                tf = len(temp)
                if(tf != 0): #len(temp) gives me the frequency of occurrence of the phrase in that doc. row gives me the doc. 
                    term_frequency[identifier] = (1+math.log10(tf))/row[1]
                else:
                    term_frequency[identifier] = 0 
    
    return term_frequency





def freetext_query(query):        # free text query
    # print(query)
    
    doc_scores = dict()
    matrix = ''
    with open("tfidf_matrix", "rb") as matrix_file:
        matrix = pickle.load(matrix_file)
    
    vectorizer = ''
    with open("tfidf_vectorizer", "rb") as vect_file:
        vectorizer = pickle.load(vect_file)
    
    doc_map = ''
    with open("document_mapping", "rb") as doc_file:
        doc_map = pickle.load(doc_file)

    query_string = ' '.join(query)
    print(query_string)
    query_vector = vectorizer.transform([query_string])
    print(np.shape(query_vector))

    doc_vectors = []

    for i in files:
        filename = i
        docs = set()

        with open(path + filename, 'rb') as infile:
            B = pickle.load(infile)

            for term in query:
                key = tree_search(B, term)
                if(key):
                    # print(posting_list)

                    no_of_items = key[1]
                    for i in range(no_of_items):
                        doc = key[2][i][0]
                        docs.add(doc)
                        
                        identifier = (filename+'.csv',doc)

                        if identifier not in doc_scores:
                            col = doc_map.index(identifier)
                            doc_vector = [matrix.iloc[col]]
                            # doc_vector = matrix[col].to_frame().T
                            print(np.shape(doc_vector))
                            doc_scores[identifier] = cosine_similarity(query_vector,doc_vector)
                            print(doc_scores[identifier])
                                    
                    # for doc in docs:
                    #     identifier = (filename+'.csv',doc)
                    #     col = doc_map.index(identifier)
                    #     row = term
                    #     score = matrix[col][row]

                    #     if identifier in doc_scores:
                    #         doc_scores[identifier] += (score*score)
                    #     else:
                    #         doc_scores[identifier] = (score*score)
                        #query -> malaysia sand dunes
                        #list -> [((MSNBC01, 2),0.534),((CNN02,4),0.765)]

    return doc_scores




def rank(input, top = 10, isphrase = False):
    if isphrase == True:
        scores = phrase_query(input)
    else:
        scores = freetext_query(input)

    df = pd.DataFrame.from_dict(scores, orient = "index",columns = ['Score'])
    #df = df['Score'].apply(lambda x: math.sqrt(x))
    df = df.sort_values('Score', ascending = False).head(top)
    return df



def fetch_snippets(identifiers):
    snippets = []
    for identifier in identifiers:
        print(identifier)
        file_addr = data_path+identifier[0]
        row_no = int(identifier[1])
        csv_file = pd.read_csv(file_addr)
        snippet = csv_file['Snippet'][row_no]
        snippets.append(snippet)
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


