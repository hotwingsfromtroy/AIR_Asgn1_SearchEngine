import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from os import listdir

from btree_implementation_mk_IV import BTree, tree_insert, print_tree, store_tree

from collections import Counter







lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) 



def construct_inverted_index():
    blocks  = listdir('./Data/TelevisionNews/')
    print(len(blocks))

# block_num = 0
    for i in blocks:
        B = BTree(8)

        try:
            csv = pd.read_csv('./Data/TelevisionNews/'+i)
        except:
            continue
        doc_id = 0
        if 'Snippet' not in csv.columns:
            continue
        for doc in csv['Snippet']:
        # print(csv['Snippet'][0])
            # print(word_tokenize(csv['Snippet'][0]))
            temp = [ lem.lemmatize(x) for x in word_tokenize(doc) if x.isalnum() and x not in stop_words ]
            temp = Counter(temp)
            for term, count in temp.items(): 
                tree_insert(B, [term, (doc_id, count)])
            
            doc_id +=1        
        # print_tree(B)
        # print(B)
        print(i, i.split('.'))
        store_tree(B, './InvertedIndex/', '.'.join(i.split('.')[:2]))
        # block_num += 1
        # if block_num==5:
        #     break
