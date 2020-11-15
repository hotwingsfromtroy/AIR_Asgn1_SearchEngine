import pickle
#from inverted_index import construct_inverted_index
from ranking import rank, fetch_snippets
from os import listdir
import pickle
import pandas as pd
import nltk
# nltk.download()
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

vocab = ''
with open('tfidf_matrix', 'rb') as infile:
    matrix = pickle.load(infile)
#         # print(matrix.columns.tolist())
    vocab_list = matrix.columns.tolist()


query = input("Enter query: ")
while(query!="exit"):
    start = time.perf_counter()

    if(query!="phrase query"):
        isphrase = False
        
    else:
        query = input("Enter phrase: ")
        isphrase = True
        
    
    temp_query_1 = [lem.lemmatize(x.lower()) for x in word_tokenize(query) if x.isalnum() and x not in stop_words]
    
    mod_query = []
    for term in temp_query_1:
        if term in vocab_list:
            mod_query.append(term)
        else:
            corrections = get_correction(term, vocab_list, 3)
            if corrections:
                mod_query.append(list(corrections)[0])
    
    print('Fetching results for query ', end = '')
    if isphrase:
        print('phrase ', end = '')
    
    print('"'+ ' '.join(mod_query)+'"')
    
    
    result = rank(mod_query, isphrase = isphrase)
    print(result)

    ids = result.index.values.tolist()
    print(ids)
    snippets = fetch_snippets(ids)
    end = time.perf_counter()
    
    for snippet in snippets:
        print(snippet)
        print()
    
    print("TIME TAKEN TO RETRIEVE SEARCH RESULTS: ", end-start)
    print()
    print()
    query = input("Enter query: ")


# df = dict ()
# df['apple'] = 3.4
# df['banana'] = 1.9
# df['cherry'] = 2.6

# df = pd.DataFrame.from_dict(df, orient = 'index', columns = ['Score'])
# print(df)
# df = df.sort_values('Score')
# print(df)



