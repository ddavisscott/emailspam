import nltk
import os
import random
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk import NaiveBayesClassifier, classify

lem = WordNetLemmatizer()
stoplist = stopwords.words('english')

def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, encoding = "ISO-8859-1")
        a_list.append(f.read())
    f.close()
    return a_list 

def preprocess(sentence):
    return [lem.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.isalpha()]

def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier

def evaluate(train_set, test_set, classifier):
    print ('Accuracy on the training set = ' + str(classify.accuracy(classifier, train_set)))
    print ('Accuracy of the test set = ' + str(classify.accuracy(classifier, test_set)))

# loading data

for i in range(5):
    spam = init_lists('enron' +str(i+1) +'/spam/')
    ham = init_lists('enron' + str(i+1)  +'/ham/')
    # combine the two lists keeping the labels
    all_emails = [(email, 'spam') for email in spam]
    all_emails += [(email, 'ham') for email in ham]




print (len(all_emails))

# shuffle the examples 
random.shuffle(all_emails)

# Label the data
all_features = [(get_features(email, 'bow'), label) for (email, label) in all_emails]
print(all_features)

# split it into a training
train_set, test_set, classifier = train(all_features, 0.8)

evaluate(train_set, test_set, classifier)
classifier.show_most_informative_features(20)
