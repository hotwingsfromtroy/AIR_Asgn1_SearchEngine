import pickle
import pandas as pd
import nltk
# nltk.download()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from os import listdir
from btree_implementation import BTree, tree_insert, print_tree, store_tree, tree_search
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize


path = './InvertedIndex/'
data_path = './Data/TelevisionNews/'
vector_path = './Vectors/'
files = listdir(path)


lem = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))



def get_tfidf_matrix(columns):
    vectorized_docs = vectorizer.fit_transform(columns)
    term_document_matrix = pd.DataFrame(vectorized_docs.T.toarray(), index=vectorizer.get_feature_names())
    return vectorizer, term_document_matrix

def position_finder(list_of_words):
    position_dict = dict()
    for i in range(len(list_of_words)):
        if list_of_words[i] not in position_dict.keys():
            position_dict[list_of_words[i]] = []
        position_dict[list_of_words[i]].append(i)
    return position_dict


def new_position_finder(temp_list):
    index = 0
    temp = {}
    for word in temp_list:
        k = lem.lemmatize(word)
        if (k in temp):
            temp[k].append(index)
        else:
            temp[k] = [index]
        index += 1
    return temp


def one_word_query(query):
    flag = 0

    for i in files:
        filename = i

        with open(path+filename, 'rb') as infile:
            B = pickle.load(infile)
            posting_list = tree_search(B, query)

            if(posting_list):
                print(posting_list)
                csv = pd.read_csv(data_path + filename + ".csv")
                no_of_items = posting_list[1]
                for i in range(no_of_items):
                    flag = 1
                    print(csv.values[posting_list[2][i][0]][6]+"\n")
    if(flag == 0):
        print('No results for query searched')





def multi_word_query(query):        # free text query
    # print(query)
    for i in files:
        filename = i
        docs = set()

        with open(path + filename, 'rb') as infile:
            B = pickle.load(infile)

            for term in query:
                posting_list = tree_search(B, term)
                if(posting_list):
                    # print(posting_list)

                    no_of_items = posting_list[1]
                    for i in range(no_of_items):
                        docs.add(posting_list[2][i][0])

        csv = pd.read_csv(data_path + filename + ".csv")
        for docID in docs:
            print(csv.values[docID][6]+"\n")


def phase_query(query):
    for i in files:
        filename = i
        docs = {}

        with open(path + filename, 'rb') as infile:
            B = pickle.load(infile)
            common = []
            for term in query:
                docIDs = set()
                posting_list = tree_search(B, term)
                if(posting_list):
                    # print(posting_list)
                    docs[term] = {}
                    no_of_items = posting_list[1]
                    for i in range(no_of_items):
                        docIDs.add(posting_list[2][i][0])
                        docs[term][posting_list[2][i][0]] = posting_list[2][i][1]
                common.append(docIDs)

            inter = common[0]               # for performing intersection
            for i in common:
                inter = inter & i

            # inside inter, we get all row numbers in one file having all the terms in the query


            for row in inter:                 # for each row
                check = []
                for term in query:
                    check.append(docs[term][row])

                temp = set(check[0])
                for i in range(1, len(check)):
                    for j in range(len(check[i])):
                        check[i][j] -= i
                    temp &= set(check[i])

                if(len(temp) != 0):
                    csv = pd.read_csv(data_path + filename + ".csv")
                    print(csv.values[row][6]+"\n")




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
            print(i)
        except:
            continue

        #doc_id based on the row number in the dataframe
        doc_id = 0

        #checking for colums with snippets
        if 'Snippet' not in csv.columns:
            continue


        vector_list = []
        #for each row in the csv file...
        for doc in csv['Snippet']:
            #the snippet is tokenized, then stop words and non-alphanumeric words are removed from this list and the remaining words are lemmatized.
            temp_1 = [ lem.lemmatize(x) for x in word_tokenize(doc) if x.isalnum() and x not in stop_words ]
            # temp2 = [lem.lemmatize(x) for x in word_tokenize(doc) ]
            # print(doc)

            #Counter makes a dictionary of words in temp as key and with their frequency as the value
            # temp = Counter(temp)
            # print(temp)
            s = ' '.join(temp_1)

            vector_list.append(s)

            # vec, matrix = get_tfidf_matrix(some)
            # print(vec)
            temp_list = doc.split(' ')
            # print(l)



            temp = new_position_finder(temp_list)


            # print(temp)


            for term, position_list in temp.items():
                #adding term, doc_id, and count to the tree
                tree_insert(B, [term, (doc_id, position_list)])
            doc_id +=1


        #pickle the constructed tree and store it in the ./InvertedIndex/ folder with the same name as the csv file
        store_tree(B, './InvertedIndex/', '.'.join(i.split('.')[:2]))

        vec, matrix = get_tfidf_matrix(vector_list)
        # print(matrix)
        # print(vec)
        with open( './Vectors/' + '.'.join(i.split('.')[:2]), 'wb') as outfile:
            pickle.dump((vec,matrix),outfile)

    print('stop')


import time

tic = time.perf_counter()
# construct_inverted_index()
toc = time.perf_counter()

print(toc-tic)












