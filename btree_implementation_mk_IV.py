import pickle

#function to initialize new Key. contains term, document frequecy and posting list
def Key(term):
    # return [term, 'doc_freq':0, 'posting_list':[]}
    return [term, 0, []]

#function to initialize B-Tree node. Contains list of keys and list of children(which are nodes)
def Node():
    # return {'keys':[], 'children':[]}
    return [[],[]]

#function to print the keys of a node. Indentation is based on the depth of the node. Then print_node function is invoked for the children of the node.
def print_node(node, tab):
    indent = ''.join(['-' for i in range(tab)])
    print(indent, end = '')
    for i in node[0]:
        print(i[0], end=' ')
    print('\n')
    for i in node[1]:
        print_node(i, tab+1)


#function to search for term in list of keys in node. Uses binary search.
#if term is found returns dictionary with 'search' = True and 'pos' = index of found key in the list of keys.
#if term is not found then return dictionary with 'search' = False and 'pos' = index of child node which might contain the term
#if node is empty then return dictionary with 'search' = False and 'pos' = -1
def node_search(node, term):
    l = 0; r = len(node[0])-1
    while(l<=r):
        mid = int((l+r)/2)
        if node[0][mid][0] == term:
            return {'search':True, 'node': node, 'pos': mid}
        elif term > node[0][mid][0]:
            if l==r:
                return {'search':False, 'node': node, 'pos': mid+1}
            else:
                l = mid+1
        else:
            if l==mid:
                return {'search':False, 'node': node, 'pos': mid}
            else:
                r = mid -1

    return {'search':False, 'node': node, 'pos': -1}






#function to add new key to leaf node. Limit is the max number of keys allowed for the node.
def node_vanilla_add(node, new_key, limit):
    node[0].append(new_key)
    node[0] = sorted(node[0], key= lambda x: x[0])
    l = len(node[0])

    #if the addition of the new key makes list longer than limit, split the node.
    #retrive new key and return it along with newly made child nodes.
    if l > limit:
        new_key = node[0][int(l/2)]
        child1 = Node()
        child1[0] = node[0][:int(l/2)]
        child2 = Node()
        child2[0] = node[0][int(l/2) + 1:]
        return {'key':new_key, 'child1':child1, 'child2':child2}
    return None


#function to add new key along with new children which are the result of adding a key to a full leaf node(fron vanilla_add or complex_add)
def node_complex_add(node, new_stuff, limit):
    node[0].append(new_stuff['key'])
    node[0] = sorted(node[0], key= lambda x: x[0])
    i = node[0].index(new_stuff['key'])

    #adding child nodes into the appropriate location in the list of children
    node[1] = node[1][:i] + [new_stuff['child1'], new_stuff['child2']] + node[1][i+1:]
    l = len(node[0])

    #if the addition of new key makes list longer than limit, split node
    #retrive new key and return it along with newly made child nodes.
    if l > limit:
        mid = int(l/2)
        child1 = Node()
        child1[0] = node[0][:mid]
        child1[1] = node[1][:mid+1]
        
        child2 = Node()
        child2[0] = node[0][mid+1:]
        child2[1] = node[1][mid+1:]

        return {'key': node[0][mid], 'child1':child1, 'child2':child2}
        
    return None

#function to initialize B-Tree. contains maximum number of keys allowed per node, number of keys present in the tree,
#number of documents(rows of the csv file) that were processed to insert into the tree, root node of the tree
def BTree(degree):
    return {'max_keys':degree-1, 'n':0, 'doc_count':0, 'root':Node()}


#function to add term and a tuple of doc_id and term frequency(w.r.t document) to the tree
def tree_insert(tree, stuff):
    term = stuff[0]
    node_res_stack = []
    temp = tree['root']
    res = ''
    while(1):
        res = node_search(temp, term)

        #adding searched node into a stack
        node_res_stack.append(res)
        if res['search']:
            #if node with term is found
            break
        else:
            #if the node is empty(no keys) or if there aren't any children(list of children is empty) then break
            if res['pos'] == -1 or res['node'][1] == []:
                break
            
            #if 'search' is false and position of child node is returned then check child node.
            temp = res['node'][1][res['pos']]

    
    temp = None

    #if the last node added to stack has the term
    if node_res_stack[-1]['search']:

        #retrieving the appropriate key from the list of keys in the node
        req_key = node_res_stack[-1]['node'][0][node_res_stack[-1]['pos']]

        #incrementing the doc_freq value
        req_key[1] += 1

        #appending (document, term frequency) tuple to the posting list
        req_key[2].append(stuff[1])
        return

    while(node_res_stack):
        #pop the top of the stack
        res = node_res_stack.pop()

        #if the last node is empty(this how it is for the root node) or has no children, i.e a leaf node
        if res['pos'] == -1 or res['node'][1] == []:
            new_key = Key(term)
            new_key[1] += 1
            new_key[2].append(stuff[1])

            #incrementing the number of keys in the tree
            tree['n'] += 1
            temp = node_vanilla_add(res['node'], new_key, tree['max_keys'])

            #if temp isn't None and the stack is empty, i.e, new root node must be made
            if temp:
                if not node_res_stack :
                    new_node = Node()
                    new_node[0].append(temp['key'])
                    new_node[1].append(temp['child1'])
                    new_node[1].append(temp['child2'])
                    tree['root'] = new_node
                    break
            else:
                break
        
            continue
            
        if temp:
            temp = node_complex_add(res['node'], temp, tree['max_keys'])

            #if temp isn't None and the stack is empty, i.e, new root node must be made
            if temp:
                if not node_res_stack:
                    new_node = Node()
                    new_node[0].append(temp['key'])
                    new_node[1].append(temp['child1'])
                    new_node[1].append(temp['child2'])
                    tree['root'] = new_node
                    break
            else:
                break

def tree_search(tree, term):
        
    temp = tree['root']
    while(1):
        res = node_search(temp, term)
        if res['search']:
            # return {'node':res['node'], 'pos':res['pos']}
            return res['node'][0][res['pos']]
        else:
            if res['pos'] == -1 or res['node'][1] == []:
                return None
            
            temp = res['node'][1][res['pos']]



def print_tree(tree):
        print_node(tree['root'], 0)


def store_tree(tree, path, filename):
    with open(path+filename, 'wb') as outfile:
        pickle.dump(tree, outfile)


# B = BTree(4)
# tree_insert(B, ['a', 1])
# tree_insert(B, ['b', 2])
# tree_insert(B, ['c', 3])
# tree_insert(B, ['d', 4])
# tree_insert(B, ['e', 4])
# tree_insert(B, ['f', 4])
# tree_insert(B, ['g', 4])
# tree_insert(B, ['h', 4])
# tree_insert(B, ['i', 4])
# tree_insert(B, ['j', 4])
# tree_insert(B, ['k', 4])
# tree_insert(B, ['l', 4])
# tree_insert(B, ['m', 4])



# tree_insert(B, ['n', 1])
# tree_insert(B, ['o', 2])
# tree_insert(B, ['p', 3])
# tree_insert(B, ['q', 4])
# tree_insert(B, ['r', 4])
# tree_insert(B, ['s', 4])
# tree_insert(B, ['t', 4])
# tree_insert(B, ['u', 4])
# tree_insert(B, ['v', 4])
# tree_insert(B, ['w', 4])
# tree_insert(B, ['x', 4])
# tree_insert(B, ['y', 4])
# tree_insert(B, ['z', 4])



# tree_insert(B, ['ab', 1])
# tree_insert(B, ['bc', 2])
# tree_insert(B, ['cd', 3])
# tree_insert(B, ['de', 4])
# tree_insert(B, ['ef', 4])
# tree_insert(B, ['fg', 4])
# tree_insert(B, ['gh', 4])
# tree_insert(B, ['hi', 4])
# tree_insert(B, ['ij', 4])
# tree_insert(B, ['jk', 4])
# tree_insert(B, ['kl', 4])
# tree_insert(B, ['lm', 4])
# tree_insert(B, ['mn', 4])


# tree_insert(B, ['no', 1])
# tree_insert(B, ['op', 2])
# tree_insert(B, ['pq', 3])
# tree_insert(B, ['qr', 4])
# tree_insert(B, ['rs', 4])
# tree_insert(B, ['st', 4])
# tree_insert(B, ['tu', 4])
# tree_insert(B, ['uv', 4])
# tree_insert(B, ['vw', 4])
# tree_insert(B, ['wx', 4])
# tree_insert(B, ['xy', 4])
# tree_insert(B, ['yz', 4])
# tree_insert(B, ['za', 4])


# tree_insert(B, ['abc', 1])
# tree_insert(B, ['bcd', 2])
# tree_insert(B, ['cde', 3])
# tree_insert(B, ['def', 4])
# tree_insert(B, ['efg', 4])
# tree_insert(B, ['fgh', 4])
# tree_insert(B, ['ghi', 4])
# tree_insert(B, ['hij', 4])
# tree_insert(B, ['ijk', 4])
# tree_insert(B, ['jkl', 4])
# tree_insert(B, ['klm', 4])
# tree_insert(B, ['lmn', 4])
# tree_insert(B, ['mno', 4])


# tree_insert(B, ['nop', 1])
# tree_insert(B, ['opq', 2])
# tree_insert(B, ['pqr', 3])
# tree_insert(B, ['qrs', 4])
# tree_insert(B, ['rst', 4])
# tree_insert(B, ['stu', 4])
# tree_insert(B, ['tuv', 4])
# tree_insert(B, ['uvw', 4])
# tree_insert(B, ['vwx', 4])
# tree_insert(B, ['wxy', 4])
# tree_insert(B, ['xyz', 4])
# tree_insert(B, ['yza', 4])
# tree_insert(B, ['zab', 4])



# tree_insert(B, ['abcd', 1])
# tree_insert(B, ['bcde', 2])
# tree_insert(B, ['cdef', 3])
# tree_insert(B, ['defg', 4])
# tree_insert(B, ['efgh', 4])
# tree_insert(B, ['fghi', 4])
# tree_insert(B, ['ghij', 4])
# tree_insert(B, ['hijk', 4])
# tree_insert(B, ['ijkl', 4])
# tree_insert(B, ['jklm', 4])
# tree_insert(B, ['klmn', 4])
# tree_insert(B, ['lmno', 4])
# tree_insert(B, ['mnop', 4])


# tree_insert(B, ['nopq', 1])
# tree_insert(B, ['opqr', 2])
# tree_insert(B, ['pqrs', 3])
# tree_insert(B, ['qrst', 4])
# tree_insert(B, ['rstu', 4])
# tree_insert(B, ['stuv', 4])
# tree_insert(B, ['tuvw', 4])
# tree_insert(B, ['uvwx', 4])
# tree_insert(B, ['vwxy', 4])
# tree_insert(B, ['wxyz', 4])
# tree_insert(B, ['xyza', 4])
# tree_insert(B, ['yzab', 4])
# tree_insert(B, ['zabc', 4])







# ######################################################








# # print_tree(B)


# ans = tree_search(B, 'abcded')
# print(ans)


# print(B['root'])

# with open('btree4', 'wb') as outfile:
#     pickle.dump(B, outfile)



