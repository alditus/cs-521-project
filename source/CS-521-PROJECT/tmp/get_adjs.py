## helper functiont o find nearest neightbor respect to extracted target phrase

tokenized_lists = [nltk.pos_tag(word_tokenize(i)) for i in dataset['statement']]

## helper function to find nearest adjective respect to target phrase
def search_adj(word_list, target, text_tag):
    """
    Search for a part of speech (POS) nearest a target phrase of interest.
    Parameters
    ----------
    word_list: list
        Sentence that's already been tokenized/POS-tagged
    target: str
        Target phrase of interest
    text_tag: str
        Nearby POS
    """

    # Find the locations of your target phrase in the sentence
    target_index = [index for index, word in enumerate(word_list) if word[0] == target]
    if len(target_index) == 0:
        return None
    else:
        target_index = target_index[0]
    for i, entry in enumerate(word_list):
        try:
            ahead = target_index + 1 + i
            behind = target_index + 1 - i
            if (word_list[ahead][1]) == text_tag:
                return word_list[ahead][0]
            elif (word_list[behind][1]) == text_tag:
                return word_list[behind][0]
            else:
                continue
        except IndexError:
            pass

for sentence in tokenized_lists:
    print(search_adj(sentence, 'the', 'JJ'))