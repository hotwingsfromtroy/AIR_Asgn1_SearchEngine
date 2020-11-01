# AIR_Asgn1_SearchEngine

Three implementations of btree made.  
Version I - user defined classes for everything and pickling for local storage. Size of test file - 7KB  
Version II - only a single user defined class, and pickling for local storage. Size of test file - 5KB  
Version III - nested dictionaries, with json as the local storage. Size of test file - 8KB  
Version IV - nested lists, with pickling(data serialization). Size of test file - 4KB

Serialization better than json. User readability isn't a concern here. We can go with nested dictionaries with pickling. Maybe another method of serialization if it turns out to be better.
Check avro and protobuff for this.  

Of course, all of this is useless if it turns out that a plain dictionary(without implementing btree) works out better. Don't know if there are optimizations to take care of cases with large number keys.
Python dictionary is implemented using hash tables. Unless our btree functions are really inefficient, it should work out better that hash table method for large data sets. We can either stick to btree and say we're keeping scalbility in mind or just work with the hash table one.


# Immediate To-Do
- reduce the number of dictionaries in the inverted index, replace with lists. A term-docid pair as a dictionary takes up around 200. List is only ~50-60. -- DONE
