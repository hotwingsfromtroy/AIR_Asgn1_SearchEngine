import pickle



def get_edit_distance(word1, word2, max_dist):
    # dp_matrix = 
    dp_prev = list(range(len(word2)+1))
    dp_curr = ''
    for i in range(len(word1)):
        dp_curr = [0 for _ in range(len(word2)+1)]
        dp_curr[0] = dp_prev[0] +1
        check = 1
        for j in range(len(word2)):
            if word1[i] == word2[j]:
                dp_curr[j+1] = dp_prev[j]
            else:
                dp_curr[j+1] = min([dp_curr[j], dp_prev[j], dp_prev[j+1]]) + 1
            if dp_curr[j+1] < max_dist:
                check = 0
        dp_prev = dp_curr
        # print(dp_curr)
        if check:
            return -1
    
    return dp_curr[-1]





# x = get_edit_distance('edwind', 'steve', 2)
# print(x)


def get_correction(word, vocabulary, max_dist):
    correction = set()
    # max_dist = len(word)
    for _word in vocabulary:
        temp = get_edit_distance(word, _word, max_dist)
        if temp == -1:
            continue
        if temp < max_dist:
            max_dist = temp
            correction = {_word}
        
        else:
            correction.add(_word)
    
    return correction


# import time
# import multiprocessing

# if __name__ == '__main__':
#     with open('tfidf_matrix', 'rb') as infile:
#         matrix = pickle.load(infile)
#         # print(matrix.columns.tolist())
#         vocab = matrix.columns.tolist()
#         print('==================')
#         # print(matrix.index)
#         query = ['poltics', 'gushe', 'holp', 'tirrany', 'bivh', 'wut', 'kelp', 'jest', 'warming', 'ridiculous', 'sectumsempra']
#         print('start')
#         start = time.perf_counter()


#         results = [get_correction(a, vocab, 2) for a in query]
        
        
        
#         for i in range(len(query)):
#             print(query[i], results[i])
#         end = time.perf_counter()
#         print(end-start)

