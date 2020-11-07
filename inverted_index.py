import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from os import listdir
from btree_implementation_mk_IV import BTree, tree_insert, print_tree, store_tree
from collections import Counter

lem = WordNetLemmatizer() 
stop_words = set(stopwords.words('english')) 


def position_finder(list_of_words):
    position_dict = dict()
    for i in range(len(list_of_words)):
        if list_of_words[i] not in position_dict.keys():
            position_dict[list_of_words[i]] = []
        position_dict[list_of_words[i]].append(i)
    return position_dict
    

#function to construct inverted index for all files in ./Data/TelevisionNews/ folder
def construct_inverted_index():
    print('start')
    #listdir gets list of files in the path
    blocks  = listdir('./Data/TelevisionNews/')
    # print(len(blocks))

# block_num = 0
    for i in blocks:

        #initialising B-Tree with degree 8. 8 was chosen arbitrarily. Can change if another degree works better.
        B = BTree(8)

        try:
            #getting dataframe of info in csv file. It is within 'try' to make sure files with no content in them don't break the code.
            csv = pd.read_csv('./Data/TelevisionNews/'+i)
        except:
            continue

        #doc_id based on the row number in the dataframe
        doc_id = 0

        #checking for colums with snippets
        if 'Snippet' not in csv.columns:
            continue

        #for each row in the csv file...
        for doc in csv['Snippet']:
            #the snippet is tokenized, then stop words and non-alphanumeric words are removed from this list and the remaining words are lemmatized.
            temp = [ lem.lemmatize(x) for x in word_tokenize(doc) if x.isalnum() and x not in stop_words ]

            #Counter makes a dictionary of words in temp as key and with their frequency as the value
            # temp = Counter(temp)
            # print(temp)
            temp = position_finder(temp)
            for term, position_list in temp.items(): 
                #adding term, doc_id, and count to the tree
                tree_insert(B, [term, (doc_id, position_list)])
            doc_id +=1        
      
        #pickle the constructed tree and store it in the ./InvertedIndex/ folder with the same name as the csv file
        store_tree(B, './InvertedIndex/', '.'.join(i.split('.')[:2]))
        # print(i)
    print('stop')


import time

tic = time.perf_counter()
construct_inverted_index()
toc = time.perf_counter()

print(toc-tic)