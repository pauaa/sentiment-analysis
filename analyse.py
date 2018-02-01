# -*- coding: utf-8 -*-

import codecs
import time

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from preprocess import preprocess

t0 = time.time()
features_train, features_test, labels_train, labels_test = preprocess()

print "pre-processing done"

# Train the classifier
clf = SVC(kernel='rbf', class_weight='balanced', C=500, gamma=0.001)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)

print "accuracy of SVC ", accuracy

# Write predictions into a file
f = codecs.open("predictions.tsv", "w")
for prediction in pred:
    if prediction == 0:
        prediction = "negative"
    elif prediction == 1:
        prediction = "neutral"
    elif prediction == 2:
        prediction = "positive"
    f.write(prediction)
    f.write(" ")
f.close()

print "time: ", time.time() - t0
