#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
import collections, numpy

sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = svm.SVC(kernel='rbf', C=10000)

print clf


clf.fit(features_train[:len(features_train)/100], labels_train[:len(features_train)/100])

result = clf.predict(features_test)

print (result == 0).sum()
print (result == 1).sum()

print accuracy_score(labels_test, result)
