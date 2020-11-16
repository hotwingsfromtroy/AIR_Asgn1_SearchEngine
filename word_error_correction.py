# Calculating the Levenshtein Distance for two words with an upper cap on the number of allowed edits.
def get_edit_distance(word1, word2, max_dist):
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
            
            # If any of the possible edits are less than the upper limit
            if dp_curr[j+1] < max_dist:
                check = 0
        dp_prev = dp_curr
   
        # Return -1 if word can't be converted to the other within the specified number of edits.
        if check:
            return -1
    
    # Return edit distance.
    return dp_curr[-1]


# Get a set of corrections for a given word from a given vocabulary.
# max_dist indicated the max number of edits allowed for the convertion.
def get_correction(word, vocabulary, max_dist):

    # Set of possible corrections.
    correction = set()
    for _word in vocabulary:
        temp = get_edit_distance(word, _word, max_dist)
        
        # If word isn't within max_dist number of edit of _word, move on.
        if temp == -1:
            continue
        
        # If the number of edits is less than the specified limit, make that the new limit.
        # Remove all previous corrections from the set.
        if temp < max_dist:
            max_dist = temp
            correction = {_word}
        
        else:
            correction.add(_word)
    
    # Return the set of corrections.
    return correction
