# -*- coding: utf-8 -*-

import time

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from preprocess import preprocess
from fscore import fscore

t0 = time.time()
features_train, features_test, labels_train, labels_test = preprocess()

print "pre-processing done"

# Train the classifier
clf = SVC(kernel='rbf', class_weight='balanced', C=500, gamma=0.001)
clf.fit(features_train, labels_train)
predicted_labels = clf.predict(features_test)
accuracy = accuracy_score(labels_test, predicted_labels)
f_score = fscore(labels_test, predicted_labels)

print "accuracy of SVC ", accuracy
print "f-score of SVC ", f_score

print "time: ", time.time() - t0
