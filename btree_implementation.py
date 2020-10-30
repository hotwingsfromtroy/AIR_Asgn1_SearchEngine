class KEY:
    def __init__(self, term):
        self.term = term
        self.doc_id_freq = 0
        self.posting_list = []



class NODE:
    def __init__(self):
        self.keys = []
        self.children = []

    
    def print_node(self, tab):
        indent = ''.join(['-' for i in range(tab)])
        print(indent, end = '')
        for i in self.keys:
            print(i.term, end=' ')
        # print('\n--------\n')
        print('\n')
        for i in self.children:
            
            i.print_node(tab+1)
            # print('\n')
    
    
    def search(self, term):

        l = 0; r = len(self.keys)-1
        while(l<=r):
            mid = int((l+r)/2)
            if self.keys[mid].term == term:
                return {'search':True, 'node': self, 'pos': mid}
            elif term > self.keys[mid].term:
                if l==r:
                    return {'search':False, 'node': self, 'pos': mid+1}
                else:
                    l = mid+1
            else:
                if l==r:
                    return {'search':False, 'node': self, 'pos': mid}
                else:
                    r = mid -1

        return {'search':False, 'node': self, 'pos': -1}
    
    

    
    
    def vanilla_add(self, new_key, limit):
        self.keys.append(new_key)
        self.keys = sorted(self.keys, key= lambda x: x.term)
        l = len(self.keys)
        if l > limit:
            new_key = self.keys[int(l/2)]
            child1 = NODE()
            child1.keys = self.keys[:int(l/2)]
            child2 = NODE()
            child2.keys = self.keys[int(l/2) + 1:]
            return {'key':new_key, 'child1':child1, 'child2':child2}
        return None

    def complex_add(self, new_stuff, limit):
        self.keys.append(new_stuff['key'])
        self.keys = sorted(self.keys, key= lambda x: x.term)
        i = self.keys.index(new_stuff['key'])
        self.children = self.children[:i] + [new_stuff['child1'], new_stuff['child2']] + self.children[i+1:]
        l = len(self.keys)
        if l > limit:
            mid = int(l/2)
            child1 = NODE()
            child1.keys = self.keys[:mid]
            child1.children = self.children[:mid+1]
            
            child2 = NODE()
            child2.keys = self.keys[mid+1:]
            child2.children = self.children[mid+1:]

            return {'key': self.keys[mid], 'child1':child1, 'child2':child2}
            
        return None
        
        


class BTREE:
    
    def __init__(self, degree):
        self.max_keys = degree-1
        self.max_children = degree
        self.n = 0
        self.root = NODE()


    def insert(self, stuff):
        self.n += 1
        term = stuff['term']
        node_stack = []
        temp = self.root
        res = ''
        while(1):
            res = temp.search(term)
            node_stack.append(res)
            if res['search']:
                # node_stack.append(res)
                break
            else:

                if res['pos'] == -1 or res['node'].children == []:
                    break
                
                temp = res['node'].children[res['pos']]

        
        temp = None

        if node_stack[-1]['search']:
            req_key = node_stack[-1]['node'].keys[node_stack[-1]['pos']]
            req_key.doc_id_freq += 1
            req_key.posting_list.append(stuff['doc_id'])
            return

        # print(node_stack)

        while(node_stack):
            node = node_stack.pop()
            if node['pos'] == -1 or node['node'].children == []:
                new_key = KEY(term)
                new_key.doc_id_freq += 1
                new_key.posting_list.append(stuff['doc_id'])

                temp = node['node'].vanilla_add(new_key, self.max_keys)

                if temp and not node_stack :
                    new_node = NODE()
                    new_node.keys.append(temp['key'])
                    new_node.children.append(temp['child1'])
                    new_node.children.append(temp['child2'])
                    self.root = new_node
                    break
    
                continue
                
            if temp:
                # node = node_stack.pop()
                temp = node['node'].complex_add(temp, self.max_keys)
                if temp and not node_stack:
                    new_node = NODE()
                    new_node.keys.append(temp['key'])
                    new_node.children.append(temp['child1'])
                    new_node.children.append(temp['child2'])
                    self.root = new_node
                    break

    def search(self, term):
        
        temp = self.root
        while(1):
            res = temp.search(term)
            if res['search']:
                return {'node':res['node'], 'pos':res['pos']}
            else:
                if res['pos'] == -1 or res['node'].children == []:
                    return None
                
                temp = res['node'].children[res['pos']]

    def print_tree(self):
        self.root.print_node(0)




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




B.print_tree()
