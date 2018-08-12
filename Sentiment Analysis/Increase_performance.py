from nltk.tokenize import word_tokenize


class Analysis:
    def __init__(self, sentence):
        self.sentence = sentence

    def dataCollection(self):
        pos = open("dataset/positiveWord.txt", 'r').read()
        neg = open("dataset/snegativeWord.txt", 'r').read()

        self.neg = list(word_tokenize(neg))
        self.pos = list(word_tokenize(pos))

    def sentCheck(self):
        self.dataCollection()
        sentCount = 0
        word = word_tokenize(self.sentence)
        for w in word:
            if w in self.pos:
                sentCount += 2

            elif w in self.neg:
                sentCount -= 1

        return sentCount


def sentiment(sentence):
    sent = Analysis(sentence)
    return sent.sentCheck()
