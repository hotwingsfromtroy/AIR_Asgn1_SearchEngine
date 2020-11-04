import pickle
from btree_implementation_mk_IV import print_tree, tree_search
from os import listdir


path = './InvertedIndex/'

files = listdir(path)

#Choose the file/index you want to search in
filename = files[0]

B = ''
with open(path+filename, 'rb') as infile:
    B = pickle.load(infile)
    print_tree(B)

#searching for the term '2016'
print(tree_search(B, '2016'))