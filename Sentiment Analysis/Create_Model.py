import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.tokenize import RegexpTokenizer, word_tokenize

# open file with around 10000 sample of negative and positive sentence
short_pos = open("dataset/positive.txt", "r").read()
short_neg = open("dataset/negative.txt", "r").read()

# Collecting all words and sentences
all_words = []
documents = []

# removing all stop words as the,a,this,ect
stop_words = set(stopwords.words('english'))

# creating list of all words in pos.txt and neg.txt
for r in short_pos.split('\n'):
    documents.append((r, "pos"))

for r in short_neg.split('\n'):
    documents.append((r, "neg"))

# Tokenizing all the words by the regx 
tokenizer = RegexpTokenizer(r'\w+')
short_pos = tokenizer.tokenize(short_pos)
short_neg = tokenizer.tokenize(short_neg)

# adding all words in list except stop words
for w in short_pos:
    if not w in stop_words:
        all_words.append(w.lower())

for w in short_neg:
    if not w in stop_words:
        all_words.append(w.lower())

print(len(all_words))

# Pickling all the words in a pickel file 
save_documents = open("pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

# creating dictionary of the accurance of the words 
all_words = nltk.FreqDist(all_words)

# slicing the list upto 12000 words
word_features = list(all_words.keys())[:12000]
print(len(word_features))

# pickling the feature in file
save_word_features = open("pickled_algos/word_features.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


# function to define feature value
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# creating feature set for all the sentences
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# creating Naive Bay classifier for sentimental analysis
classifier = nltk.NaiveBayesClassifier.train(featuresets)
classifier.show_most_informative_features(15)

# saving Naive Bay
save_classifier = open("pickled_algos/originalnaivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# creating Multinomial Classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(featuresets)
print("mnb")
# saving MNB classifier
save_classifier = open("pickled_algos/MNB_classifier.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

# creating BernoulliNB classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(featuresets)
print("ber")
# saving BernoulliNB classifier
save_classifier = open("pickled_algos/BernoulliNB_classifier.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

# creating LogisticRegression classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(featuresets)
print("logi")
# saving LogisticRegression classifier
save_classifier = open("pickled_algos/LogisticRegression_classifier.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

# creating LinearSVC classifier
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(featuresets)
print("linear")
# saving LinearSVC classifier
save_classifier = open("pickled_algos/LinearSVC_classifier.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

# creating NuSVC classifier
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(featuresets)
print("mnb")
# saving NuSVC classifier
save_classifier = open("pickled_algos/NuSVC_classifier.pickle", "wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

# creating SGDClassifier classifier
SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(featuresets)

# saving SGDC classifier
save_classifier = open("pickled_algos/SGDC_classifier.pickle", "wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()
print("work done")
