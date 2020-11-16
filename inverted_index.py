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
from nltk.tokenize import word_tokenize
import numpy as np

# Path to folder containing inverted indices.
path = './InvertedIndex/'

# Path to folder containing raw data files.
data_path = './Data/TelevisionNews/'

# Retrieving list of names of raw data files.
files = listdir(path)

lem = WordNetLemmatizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Function to get dictionary with terms as keys and list of positions as values
def position_finder(list_of_words):
    position_dict = {}
    for pos in range(len(list_of_words)):
        if (list_of_words[pos] in position_dict):
            position_dict[list_of_words[pos]].append(pos)
        else:
            position_dict[list_of_words[pos]] = [pos]
        
    return position_dict


# Find the tf-idf matrix for all the terms and documents in the corpus.
def get_tfidf_matrix(dataset):
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    vectorized_docs = vectorizer.fit_transform(dataset)
    names = vectorizer.get_feature_names()
    tfidf_matrix = pd.DataFrame.sparse.from_spmatrix(vectorized_docs)
    tfidf_matrix.columns = names
    return tfidf_matrix, vectorizer



# Construct individual inverted indices for all files in ./Data/TelevisionNews/
def construct_inverted_index():
    
    # Get the list of files in the path.
    blocks  = listdir(data_path)

    for i in blocks:
        
        # Initialise B-Tree with degree 8 (chosen arbitrarily).
        B = BTree(8)

        try:
            # Get the dataframe of info in csv file. 
            csv = pd.read_csv(data_path+i)
        except:
            continue

        # Row number in the dataframe.
        doc_id = 0

        # Check for columns with snippets
        if 'Snippet' not in csv.columns:
            continue

        print('Started construction of inverted index for block ',i)

        # Add documents in the file to its BTree.
        for doc in csv['Snippet']:
            # The snippet is tokenized, then stop words and non-alphanumeric words are removed from this list and the remaining words are lemmatized.
            temp_1 = [ lem.lemmatize(x) for x in word_tokenize(doc) if x.isalnum() and x not in stop_words ]
            
            # temp2 = [lem.lemmatize(x) for x in word_tokenize(doc) ]
            # print(doc)

            s = ' '.join(temp_1)

            doc_list.append(s)
            doc_map.append((i,doc_id))
        
            # Getting dictionary with term and their positions in temp_1        
            temp = position_finder(temp_1)
            no_of_terms = len(temp_1)

            for term, position_list in temp.items():
                #adding term, doc_id, list of positions and number of terms to the tree
                tree_insert(B, [term, (doc_id, position_list, no_of_terms)])
            doc_id +=1


        #pickle the constructed tree and store it in the ./InvertedIndex/ folder with the same name as the csv file
        store_tree(B, './InvertedIndex/', '.'.join(i.split('.')[:2]))
        
        print('Finished construction of inverted index for block ',i)


def construct_matrix():

    #Store the mapping of the unique document identifiers to their tf-idf indices.
    with open("document_mapping","wb") as outfile:
        pickle.dump(doc_map, outfile)

    print("Started matrix construction.")
    #Get the tf-idf matrix.
    matrix, vect = get_tfidf_matrix(doc_list)
    print("Finished matrix construction.")

    #Store the tf-idf matrix.
    with open("tfidf_matrix","wb") as outfile:
        pickle.dump(matrix, outfile)

    #Store the tf-idf vectorizer.
    with open("tfidf_vectorizer","wb") as outfile:
        pickle.dump(vect, outfile)


#List containing the mapping of the unique document identifiers to their tf-idf matrix indices.
doc_map = []
doc_list = []

print("STARTED PRE-PROCESSING.")
construct_inverted_index()
construct_matrix()
print("FINISHED PRE-PROCESSING.")










