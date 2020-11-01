# AIR_Asgn1_SearchEngine

Three implementations of btree made.

Version I - user defined classes for everything and pickling for local storage. Size of test file - 7KB

Version II - only a single user defined class, and pickling for local storage. Size of test file - 5KB

Version III - nested dictionaries, with json as the local storage. Size of test file - 8KB

Serialization better than json. User readability isn't a concern here. We can go with nested dictionaries with pickling. Maybe another method of serialization if it turns out to be better.
Check avro and protobuff for this.


Of course, all of this is useless if it turns out that a plain dictionary(without implementing btree) works out better. Don't know if there are optimizations to take care of cases with large number keys.
