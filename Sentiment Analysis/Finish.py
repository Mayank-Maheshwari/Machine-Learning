import sentiment_mod as s
import increase_performance as a
from nltk.tokenize import sent_tokenize, word_tokenize


def checkSentement(string):
    result = 0
    sentence = sent_tokenize(string)
    sentence = list(sentence)
    for i in range(len(sentence)):
        sent, count = s.sentiment(sentence[i])
        if sent == 'neg':
            count = count * -1
        counter = a.sentiment(sentence[i])
        result += (2 * count) + counter
    if result > -1:
        sent = 1
    else:
        sent = -1
    return sent


def start():
    s = input("enter a string  : ")
    print(checkSentement(s))


if __name__ == "__main__":
    start()
