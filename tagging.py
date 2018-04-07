import numpy as np
from matplotlib import pyplot as plt
import csv
import random
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk import metrics
from nltk.metrics.scores import (precision, recall)
import collections


def NBtrain():
    f = open('pos-eng-5000.data.csv')
    csv_f = csv.reader(f)
    next(csv_f, None)  #skip the header line
    totalLine = sum(1 for row in csv_f)

    f = open('pos-eng-5000.data.csv')
    csv_f = csv.reader(f)
    next(csv_f, None)  #skip the header line

    trainALL = []
    testALL = []
    label = []
    testResult = []
    tt = []

    splitLine = totalLine*0.8
    line = 0

    for row in csv_f:

        if(line <= splitLine):
            trainALL.append(({'a1': row[0], 'a2': row[1], 'a3': row[2],'a4': row[3], 'a5': row[4], 'a6': row[5], 'a7': row[6]}, row[7]))
        else:
            testALL.append({'a1': row[0], 'a2': row[1], 'a3': row[2],'a4': row[3], 'a5': row[4], 'a6': row[5], 'a7': row[6]})
            label.append(row[7])
            tt.append(({'a1': row[0], 'a2': row[1], 'a3': row[2],'a4': row[3], 'a5': row[4], 'a6': row[5], 'a7': row[6]}, row[7]))
        line = line+1


    classifier = nltk.NaiveBayesClassifier.train(trainALL)
    for row in testALL:
        observed = classifier.classify(row)
        testResult.append(observed)

    accrAll = nltk.classify.accuracy(classifier, tt)
#    precAll = nltk.metrics.scores.precision(classifier, tt)
    print("The accuracy of the Naive Bayes model is {}".format(accrAll))
#    print(precAll)



def DTtrain():
    f = open('pos-eng-5000.data.csv')
    csv_f = csv.reader(f)
    next(csv_f, None)  #skip the header line
    totalLine = sum(1 for row in csv_f)

    f = open('pos-eng-5000.data.csv')
    csv_f = csv.reader(f)
    next(csv_f, None)  #skip the header line

    trainALL = []
    testALL = []
    label = []
    testResult = []
    tt = []

    splitLine = totalLine*0.8
    line = 0

    for row in csv_f:

        if(line <= splitLine):
            trainALL.append(({'a1': row[0], 'a2': row[1], 'a3': row[2],'a4': row[3], 'a5': row[4], 'a6': row[5], 'a7': row[6]}, row[7]))
        else:
            testALL.append({'a1': row[0], 'a2': row[1], 'a3': row[2],'a4': row[3], 'a5': row[4], 'a6': row[5], 'a7': row[6]})
            label.append(row[7])
            tt.append(({'a1': row[0], 'a2': row[1], 'a3': row[2],'a4': row[3], 'a5': row[4], 'a6': row[5], 'a7': row[6]}, row[7]))
        line = line+1

    classifier = nltk.DecisionTreeClassifier.train(trainALL)
    for row in testALL:
        observed = classifier.classify(row)
        testResult.append(observed)

    accrAll = nltk.classify.accuracy(classifier, tt)
#    precAll = nltk.metrics.scores.precision(classifier, tt)
    print("The accuracy of the Decision Tree model is {}".format(accrAll))


def main():
    NBtrain()
    DTtrain()


if __name__ == "__main__":
    main()
