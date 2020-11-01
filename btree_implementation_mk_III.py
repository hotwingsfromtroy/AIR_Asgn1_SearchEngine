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




def BTree(degree):
    return {'max_keys':degree-1, 'n':0, 'root':Node()}



def tree_insert(tree, stuff):
    tree['n'] += 1
    term = stuff['term']
    node_stack = []
    temp = tree['root']
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

            temp = node_vanilla_add(node['node'], new_key, tree['max_keys'])

            if temp and not node_stack :
                new_node = Node()
                new_node['keys'].append(temp['key'])
                new_node['children'].append(temp['child1'])
                new_node['children'].append(temp['child2'])
                tree['root'] = new_node
                break

            continue
            
        if temp:
            # node = node_stack.pop()
            temp = node_complex_add(node['node'], temp, tree['max_keys'])
            if temp and not node_stack:
                new_node = Node()
                new_node['keys'].append(temp['key'])
                new_node['children'].append(temp['child1'])
                new_node['children'].append(temp['child2'])
                tree['root'] = new_node
                break


def tree_search(tree, term):
        
    temp = tree['root']
    while(1):
        res = node_search(temp, term)
        if res['search']:
            return {'node':res['node'], 'pos':res['pos']}
        else:
            if res['pos'] == -1 or res['node']['children'] == []:
                return None
            
            temp = res['node']['children'][res['pos']]



def print_tree(tree):
        print_node(tree['root'], 0)






B = BTree(4)
tree_insert(B, {'term':'a', 'doc_id':1})
tree_insert(B, {'term':'b', 'doc_id':2})
tree_insert(B, {'term':'c', 'doc_id':3})
tree_insert(B, {'term':'d', 'doc_id':4})
tree_insert(B, {'term':'e', 'doc_id':4})
tree_insert(B, {'term':'f', 'doc_id':4})
tree_insert(B, {'term':'g', 'doc_id':4})
tree_insert(B, {'term':'h', 'doc_id':4})
tree_insert(B, {'term':'i', 'doc_id':4})
tree_insert(B, {'term':'j', 'doc_id':4})
tree_insert(B, {'term':'k', 'doc_id':4})
tree_insert(B, {'term':'l', 'doc_id':4})
tree_insert(B, {'term':'m', 'doc_id':4})



tree_insert(B, {'term':'n', 'doc_id':1})
tree_insert(B, {'term':'o', 'doc_id':2})
tree_insert(B, {'term':'p', 'doc_id':3})
tree_insert(B, {'term':'q', 'doc_id':4})
tree_insert(B, {'term':'r', 'doc_id':4})
tree_insert(B, {'term':'s', 'doc_id':4})
tree_insert(B, {'term':'t', 'doc_id':4})
tree_insert(B, {'term':'u', 'doc_id':4})
tree_insert(B, {'term':'v', 'doc_id':4})
tree_insert(B, {'term':'w', 'doc_id':4})
tree_insert(B, {'term':'x', 'doc_id':4})
tree_insert(B, {'term':'y', 'doc_id':4})
tree_insert(B, {'term':'z', 'doc_id':4})



tree_insert(B, {'term':'ab', 'doc_id':1})
tree_insert(B, {'term':'bc', 'doc_id':2})
tree_insert(B, {'term':'cd', 'doc_id':3})
tree_insert(B, {'term':'de', 'doc_id':4})
tree_insert(B, {'term':'ef', 'doc_id':4})
tree_insert(B, {'term':'fg', 'doc_id':4})
tree_insert(B, {'term':'gh', 'doc_id':4})
tree_insert(B, {'term':'hi', 'doc_id':4})
tree_insert(B, {'term':'ij', 'doc_id':4})
tree_insert(B, {'term':'jk', 'doc_id':4})
tree_insert(B, {'term':'kl', 'doc_id':4})
tree_insert(B, {'term':'lm', 'doc_id':4})
tree_insert(B, {'term':'mn', 'doc_id':4})


tree_insert(B, {'term':'no', 'doc_id':1})
tree_insert(B, {'term':'op', 'doc_id':2})
tree_insert(B, {'term':'pq', 'doc_id':3})
tree_insert(B, {'term':'qr', 'doc_id':4})
tree_insert(B, {'term':'rs', 'doc_id':4})
tree_insert(B, {'term':'st', 'doc_id':4})
tree_insert(B, {'term':'tu', 'doc_id':4})
tree_insert(B, {'term':'uv', 'doc_id':4})
tree_insert(B, {'term':'vw', 'doc_id':4})
tree_insert(B, {'term':'wx', 'doc_id':4})
tree_insert(B, {'term':'xy', 'doc_id':4})
tree_insert(B, {'term':'yz', 'doc_id':4})
tree_insert(B, {'term':'za', 'doc_id':4})


tree_insert(B, {'term':'abc', 'doc_id':1})
tree_insert(B, {'term':'bcd', 'doc_id':2})
tree_insert(B, {'term':'cde', 'doc_id':3})
tree_insert(B, {'term':'def', 'doc_id':4})
tree_insert(B, {'term':'efg', 'doc_id':4})
tree_insert(B, {'term':'fgh', 'doc_id':4})
tree_insert(B, {'term':'ghi', 'doc_id':4})
tree_insert(B, {'term':'hij', 'doc_id':4})
tree_insert(B, {'term':'ijk', 'doc_id':4})
tree_insert(B, {'term':'jkl', 'doc_id':4})
tree_insert(B, {'term':'klm', 'doc_id':4})
tree_insert(B, {'term':'lmn', 'doc_id':4})
tree_insert(B, {'term':'mno', 'doc_id':4})


tree_insert(B, {'term':'nop', 'doc_id':1})
tree_insert(B, {'term':'opq', 'doc_id':2})
tree_insert(B, {'term':'pqr', 'doc_id':3})
tree_insert(B, {'term':'qrs', 'doc_id':4})
tree_insert(B, {'term':'rst', 'doc_id':4})
tree_insert(B, {'term':'stu', 'doc_id':4})
tree_insert(B, {'term':'tuv', 'doc_id':4})
tree_insert(B, {'term':'uvw', 'doc_id':4})
tree_insert(B, {'term':'vwx', 'doc_id':4})
tree_insert(B, {'term':'wxy', 'doc_id':4})
tree_insert(B, {'term':'xyz', 'doc_id':4})
tree_insert(B, {'term':'yza', 'doc_id':4})
tree_insert(B, {'term':'zab', 'doc_id':4})



tree_insert(B, {'term':'abcd', 'doc_id':1})
tree_insert(B, {'term':'bcde', 'doc_id':2})
tree_insert(B, {'term':'cdef', 'doc_id':3})
tree_insert(B, {'term':'defg', 'doc_id':4})
tree_insert(B, {'term':'efgh', 'doc_id':4})
tree_insert(B, {'term':'fghi', 'doc_id':4})
tree_insert(B, {'term':'ghij', 'doc_id':4})
tree_insert(B, {'term':'hijk', 'doc_id':4})
tree_insert(B, {'term':'ijkl', 'doc_id':4})
tree_insert(B, {'term':'jklm', 'doc_id':4})
tree_insert(B, {'term':'klmn', 'doc_id':4})
tree_insert(B, {'term':'lmno', 'doc_id':4})
tree_insert(B, {'term':'mnop', 'doc_id':4})


tree_insert(B, {'term':'nopq', 'doc_id':1})
tree_insert(B, {'term':'opqr', 'doc_id':2})
tree_insert(B, {'term':'pqrs', 'doc_id':3})
tree_insert(B, {'term':'qrst', 'doc_id':4})
tree_insert(B, {'term':'rstu', 'doc_id':4})
tree_insert(B, {'term':'stuv', 'doc_id':4})
tree_insert(B, {'term':'tuvw', 'doc_id':4})
tree_insert(B, {'term':'uvwx', 'doc_id':4})
tree_insert(B, {'term':'vwxy', 'doc_id':4})
tree_insert(B, {'term':'wxyz', 'doc_id':4})
tree_insert(B, {'term':'xyza', 'doc_id':4})
tree_insert(B, {'term':'yzab', 'doc_id':4})
tree_insert(B, {'term':'zabc', 'doc_id':4})







######################################################








print_tree(B)



with open('btree3', 'w') as outfile:
    json.dump(B, outfile)



