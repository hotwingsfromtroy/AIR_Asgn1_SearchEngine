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
from nltk.tokenize import word_tokenize

path = './InvertedIndex/'
data_path = './Data/TelevisionNews/'
vector_path = './Vectors/'
files = listdir(path)

lem = WordNetLemmatizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))



def position_finder(list_of_words):
    index = 0
    temp = {}
    for word in list_of_words:
        k = lem.lemmatize(word)
        if (k in temp):
            temp[k].append(index)
        else:
            temp[k] = [index]
        index += 1
    return temp


def get_tfidf_matrix(dataset):
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    vectorized_docs = vectorizer.fit_transform(dataset)
    print(vectorized_docs)
    names = vectorizer.get_feature_names()
    tfidf_matrix = pd.DataFrame.sparse.from_spmatrix(vectorized_docs).T
    tfidf_matrix.index = names
    return tfidf_matrix


#function to construct inverted index for all files in ./Data/TelevisionNews/ folder
def construct_inverted_index():
    print('start')
    #listdir gets list of files in the path
    blocks  = listdir('./Data/TelevisionNews/')
    # print(len(blocks))

    doc_map = []
# block_num = 0
    doc_list = []
    for i in blocks:
        #initialising B-Tree with degree 8. 8 was chosen arbitrarily. Can change if another degree works better.
        B = BTree(8)

        try:
            #getting dataframe of info in csv file. It is within 'try' to make sure files with no content in them don't break the code.
            csv = pd.read_csv('./Data/TelevisionNews/'+i)
            #print(i)
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
            temp_1 = [ lem.lemmatize(x) for x in word_tokenize(doc) if x.isalnum() and x not in stop_words ]
            # temp2 = [lem.lemmatize(x) for x in word_tokenize(doc) ]
            # print(doc)

            #Counter makes a dictionary of words in temp as key and with their frequency as the value
            # temp = Counter(temp)
            s = ' '.join(temp_1)

            doc_list.append(s)
            doc_map.append((i,doc_id))
        
            temp_list = doc.split(' ')
        
            temp = position_finder(temp_list)

            for term, position_list in temp.items():
                #adding term, doc_id, and count to the tree
                tree_insert(B, [term, (doc_id, position_list)])
            doc_id +=1


        #pickle the constructed tree and store it in the ./InvertedIndex/ folder with the same name as the csv file
        store_tree(B, './InvertedIndex/', '.'.join(i.split('.')[:2]))

    with open("document_mapping","wb") as outfile:
        pickle.dump(doc_map, outfile)

    matrix = get_tfidf_matrix(doc_list)
    #print(matrix)
    print("constructed tfidf matrix")
    with open("tfidf_matrix","wb") as outfile:
        pickle.dump(matrix, outfile)
        print("Dumped matrix.")

    print('stop')



# d = [   
#         "this is some trial text",
#         "i am trying to figure out how to use some tfidf modules in python",
#         "it seems like it would be easier to write that code myself",
#         "i am not entirely sure what sort of sentences to give to accurately judge how it works",
#         "guess i will just adjust the sentences based on my requirements"
#     ]

construct_inverted_index()









