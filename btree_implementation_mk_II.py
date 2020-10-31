import json
import pickle

        

def Key(term):
    return {'term':term, 'doc_id_freq':0, 'posting_list':[]}



def Node():
    return {'keys':[], 'children':[]}


def print_node(node, tab):
    indent = ''.join(['-' for i in range(tab)])
    print(indent, end = '')
    for i in node['keys']:
        print(i['term'], end=' ')
    # print('\n--------\n')
    print('\n')
    for i in node['children']:
        
        print_node(i, tab+1)
        # print('\n')



def node_search(node, term):

    l = 0; r = len(node['keys'])-1
    while(l<=r):
        mid = int((l+r)/2)
        if node['keys'][mid]['term'] == term:
            return {'search':True, 'node': node, 'pos': mid}
        elif term > node['keys'][mid]['term']:
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







def node_vanilla_add(node, new_key, limit):
    node['keys'].append(new_key)
    node['keys'] = sorted(node['keys'], key= lambda x: x['term'])
    l = len(node['keys'])
    if l > limit:
        new_key = node['keys'][int(l/2)]
        child1 = Node()
        child1['keys'] = node['keys'][:int(l/2)]
        child2 = Node()
        child2['keys'] = node['keys'][int(l/2) + 1:]
        return {'key':new_key, 'child1':child1, 'child2':child2}
    return None

def node_complex_add(node, new_stuff, limit):
    node['keys'].append(new_stuff['key'])
    node['keys'] = sorted(node['keys'], key= lambda x: x['term'])
    i = node['keys'].index(new_stuff['key'])
    node['children'] = node['children'][:i] + [new_stuff['child1'], new_stuff['child2']] + node['children'][i+1:]
    l = len(node['keys'])
    if l > limit:
        mid = int(l/2)
        child1 = Node()
        child1['keys'] = node['keys'][:mid]
        child1['children'] = node['children'][:mid+1]
        
        child2 = Node()
        child2['keys'] = node['keys'][mid+1:]
        child2['children'] = node['children'][mid+1:]

        return {'key': node['keys'][mid], 'child1':child1, 'child2':child2}
        
    return None




# def BTree(degree):
#     return {'max_keys':degree-1, 'n':0, 'root':Node()}



# def tree_insert(tree, stuff):
#     tree['n'] += 1
#     term = stuff['term']
#     node_stack = []
#     temp = tree['root']
#     res = ''
#     while(1):
#         res = node_search(temp, term)
#         node_stack.append(res)
#         if res['search']:
#             # node_stack.append(res)
#             break
#         else:

#             if res['pos'] == -1 or res['node']['children'] == []:
#                 break
            
#             # print(res, res['node'].keys, res['node'].children)
#             temp = res['node']['children'][res['pos']]

    
#     temp = None

#     if node_stack[-1]['search']:
#         req_key = node_stack[-1]['node']['keys'][node_stack[-1]['pos']]
#         req_key['doc_id_freq'] += 1
#         req_key['posting_list'].append(stuff['doc_id'])
#         return

#     # print(node_stack)

#     while(node_stack):
#         node = node_stack.pop()
#         if node['pos'] == -1 or node['node']['children'] == []:
#             new_key = Key(term)
#             new_key['doc_id_freq'] += 1
#             new_key['posting_list'].append(stuff['doc_id'])

#             temp = node_vanilla_add(node['node'], new_key, tree['max_keys'])

#             if temp and not node_stack :
#                 new_node = Node()
#                 new_node['keys'].append(temp['key'])
#                 new_node['children'].append(temp['child1'])
#                 new_node['children'].append(temp['child2'])
#                 tree['root'] = new_node
#                 break

#             continue
            
#         if temp:
#             # node = node_stack.pop()
#             temp = node_complex_add(node['node'], temp, tree['max_keys'])
#             if temp and not node_stack:
#                 new_node = Node()
#                 new_node['keys'].append(temp['key'])
#                 new_node['children'].append(temp['child1'])
#                 new_node['children'].append(temp['child2'])
#                 tree['root'] = new_node
#                 break


# def tree_search(tree, term):
        
#     temp = tree['root']
#     while(1):
#         res = node_search(temp, term)
#         if res['search']:
#             return {'node':res['node'], 'pos':res['pos']}
#         else:
#             if res['pos'] == -1 or res['node']['children'] == []:
#                 return None
            
#             temp = res['node']['children'][res['pos']]



# def print_tree(tree):
#         print_node(tree['root'], 0)




class BTREE:
    
    def __init__(self, degree):
        self.max_keys = degree-1
        self.max_children = degree
        self.n = 0
        self.root = Node()


    def insert(self, stuff):
        self.n += 1
        term = stuff['term']
        node_stack = []
        temp = self.root
        res = ''
        while(1):
            res = node_search(temp, term)
            node_stack.append(res)
            if res['search']:
                # node_stack.append(res)
                break
            else:

                if res['pos'] == -1 or res['node']['children'] == []:
                    break
                
                # print(res, res['node'].keys, res['node'].children)
                temp = res['node']['children'][res['pos']]

        
        temp = None

        if node_stack[-1]['search']:
            req_key = node_stack[-1]['node']['keys'][node_stack[-1]['pos']]
            req_key['doc_id_freq'] += 1
            req_key['posting_list'].append(stuff['doc_id'])
            return

        # print(node_stack)

        while(node_stack):
            node = node_stack.pop()
            if node['pos'] == -1 or node['node']['children'] == []:
                new_key = Key(term)
                new_key['doc_id_freq'] += 1
                new_key['posting_list'].append(stuff['doc_id'])

                temp = node_vanilla_add(node['node'], new_key, self.max_keys)

                if temp and not node_stack :
                    new_node = Node()
                    new_node['keys'].append(temp['key'])
                    new_node['children'].append(temp['child1'])
                    new_node['children'].append(temp['child2'])
                    self.root = new_node
                    break
    
                continue
                
            if temp:
                # node = node_stack.pop()
                temp = node_complex_add(node['node'], temp, self.max_keys)
                if temp and not node_stack:
                    new_node = Node()
                    new_node['keys'].append(temp['key'])
                    new_node['children'].append(temp['child1'])
                    new_node['children'].append(temp['child2'])
                    self.root = new_node
                    break

    def search(self, term):
        
        temp = self.root
        while(1):
            res = node_search(temp, term)
            if res['search']:
                return {'node':res['node'], 'pos':res['pos']}
            else:
                if res['pos'] == -1 or res['node']['children'] == []:
                    return None
                
                temp = res['node']['children'][res['pos']]

    def print_tree(self):
        print_node(self.root, 0)

    
    def get_inverted_index(self):
        return {
            'root' : self.root
        }




B = BTREE(4)
B.insert({'term':'a', 'doc_id':1})
B.insert({'term':'b', 'doc_id':2})
B.insert({'term':'c', 'doc_id':3})
B.insert({'term':'d', 'doc_id':4})
B.insert({'term':'e', 'doc_id':4})
B.insert({'term':'f', 'doc_id':4})
B.insert({'term':'g', 'doc_id':4})
B.insert({'term':'h', 'doc_id':4})
B.insert({'term':'i', 'doc_id':4})
B.insert({'term':'j', 'doc_id':4})
B.insert({'term':'k', 'doc_id':4})
B.insert({'term':'l', 'doc_id':4})
B.insert({'term':'m', 'doc_id':4})



B.insert({'term':'n', 'doc_id':1})
B.insert({'term':'o', 'doc_id':2})
B.insert({'term':'p', 'doc_id':3})
B.insert({'term':'q', 'doc_id':4})
B.insert({'term':'r', 'doc_id':4})
B.insert({'term':'s', 'doc_id':4})
B.insert({'term':'t', 'doc_id':4})
B.insert({'term':'u', 'doc_id':4})
B.insert({'term':'v', 'doc_id':4})
B.insert({'term':'w', 'doc_id':4})
B.insert({'term':'x', 'doc_id':4})
B.insert({'term':'y', 'doc_id':4})
B.insert({'term':'z', 'doc_id':4})



B.insert({'term':'ab', 'doc_id':1})
B.insert({'term':'bc', 'doc_id':2})
B.insert({'term':'cd', 'doc_id':3})
B.insert({'term':'de', 'doc_id':4})
B.insert({'term':'ef', 'doc_id':4})
B.insert({'term':'fg', 'doc_id':4})
B.insert({'term':'gh', 'doc_id':4})
B.insert({'term':'hi', 'doc_id':4})
B.insert({'term':'ij', 'doc_id':4})
B.insert({'term':'jk', 'doc_id':4})
B.insert({'term':'kl', 'doc_id':4})
B.insert({'term':'lm', 'doc_id':4})
B.insert({'term':'mn', 'doc_id':4})


B.insert({'term':'no', 'doc_id':1})
B.insert({'term':'op', 'doc_id':2})
B.insert({'term':'pq', 'doc_id':3})
B.insert({'term':'qr', 'doc_id':4})
B.insert({'term':'rs', 'doc_id':4})
B.insert({'term':'st', 'doc_id':4})
B.insert({'term':'tu', 'doc_id':4})
B.insert({'term':'uv', 'doc_id':4})
B.insert({'term':'vw', 'doc_id':4})
B.insert({'term':'wx', 'doc_id':4})
B.insert({'term':'xy', 'doc_id':4})
B.insert({'term':'yz', 'doc_id':4})
B.insert({'term':'za', 'doc_id':4})


B.insert({'term':'abc', 'doc_id':1})
B.insert({'term':'bcd', 'doc_id':2})
B.insert({'term':'cde', 'doc_id':3})
B.insert({'term':'def', 'doc_id':4})
B.insert({'term':'efg', 'doc_id':4})
B.insert({'term':'fgh', 'doc_id':4})
B.insert({'term':'ghi', 'doc_id':4})
B.insert({'term':'hij', 'doc_id':4})
B.insert({'term':'ijk', 'doc_id':4})
B.insert({'term':'jkl', 'doc_id':4})
B.insert({'term':'klm', 'doc_id':4})
B.insert({'term':'lmn', 'doc_id':4})
B.insert({'term':'mno', 'doc_id':4})


B.insert({'term':'nop', 'doc_id':1})
B.insert({'term':'opq', 'doc_id':2})
B.insert({'term':'pqr', 'doc_id':3})
B.insert({'term':'qrs', 'doc_id':4})
B.insert({'term':'rst', 'doc_id':4})
B.insert({'term':'stu', 'doc_id':4})
B.insert({'term':'tuv', 'doc_id':4})
B.insert({'term':'uvw', 'doc_id':4})
B.insert({'term':'vwx', 'doc_id':4})
B.insert({'term':'wxy', 'doc_id':4})
B.insert({'term':'xyz', 'doc_id':4})
B.insert({'term':'yza', 'doc_id':4})
B.insert({'term':'zab', 'doc_id':4})



B.insert({'term':'abcd', 'doc_id':1})
B.insert({'term':'bcde', 'doc_id':2})
B.insert({'term':'cdef', 'doc_id':3})
B.insert({'term':'defg', 'doc_id':4})
B.insert({'term':'efgh', 'doc_id':4})
B.insert({'term':'fghi', 'doc_id':4})
B.insert({'term':'ghij', 'doc_id':4})
B.insert({'term':'hijk', 'doc_id':4})
B.insert({'term':'ijkl', 'doc_id':4})
B.insert({'term':'jklm', 'doc_id':4})
B.insert({'term':'klmn', 'doc_id':4})
B.insert({'term':'lmno', 'doc_id':4})
B.insert({'term':'mnop', 'doc_id':4})


B.insert({'term':'nopq', 'doc_id':1})
B.insert({'term':'opqr', 'doc_id':2})
B.insert({'term':'pqrs', 'doc_id':3})
B.insert({'term':'qrst', 'doc_id':4})
B.insert({'term':'rstu', 'doc_id':4})
B.insert({'term':'stuv', 'doc_id':4})
B.insert({'term':'tuvw', 'doc_id':4})
B.insert({'term':'uvwx', 'doc_id':4})
B.insert({'term':'vwxy', 'doc_id':4})
B.insert({'term':'wxyz', 'doc_id':4})
B.insert({'term':'xyza', 'doc_id':4})
B.insert({'term':'yzab', 'doc_id':4})
B.insert({'term':'zabc', 'doc_id':4})







######################################################








B.print_tree()



with open('btree2', 'wb') as outfile:
    pickle.dump(B, outfile)


