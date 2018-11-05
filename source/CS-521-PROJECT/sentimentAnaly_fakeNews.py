##    below function help us to run simple sentiment analysis on fakenews statement
##  IDEA: below code help us to list out the topics that what is mainly about fake news
##         and find out the sentiment of those topics in the fake news statement

## possible extended idea: extract all nouns/verbs/adj/adv.. in the sentence and run sentiment analysis
##                          on each sentences which has a noun/verbs/adj/adv

from textblob import TextBlob
import sys

def main():
    blob = TextBlob(sys.argv[1])
    tokens = list(blob.words)
    word, sent=[],[]
    c ,j = 0,0
    for words, pos in blob.tags:
        if pos == 'JJ' or pos == 'NN' or pos == 'JJR' or pos == 'NNS':
            word.append(words)
    if len(word) >= 2:
        for i in range(len(word)):
            if len(word) >= 2:
                print(i)
                firstw = word[0]
                secw = word[1]
                word.remove(firstw)
                word.remove(secw)
                findx = tokens.index(firstw)
                lindx = tokens.index(secw)
                sent.append(' '.join(tokens[findx:lindx + 1]))

    print(sent, tokens)
    print("Sentence and polarity")
    for sentence in sent:
        print(sentence, TextBlob(sentence).polarity)


if __name__ == '__main__':
    main()