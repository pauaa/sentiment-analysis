# -*- coding: utf-8 -*-

import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from nltk.stem.snowball import SnowballStemmer


def stem(stemmer, word):
    # stem the given word
    return stemmer.stem(word.lower())


def normalize_labels(labels):
    # convert labels to numbers
    lookup = {
        "positive": 2,
        "neutral": 1,
        "negative": 0
    }
    labels = [lookup[l] for l in labels]
    return labels


def split_labels_features(data):
    # handle and split data into features and labels
    features = [line.split('\t')[3] for line in data]
    labels = [line.split('\t')[2] for line in data]
    labels = normalize_labels(labels)

    reduced_labels, reduced_features = [], []
    for label, feature in zip(labels, features):
        if label in {0, 2}:
            reduced_labels.append(label)
            reduced_features.append(feature)
    labels, features = reduced_labels, reduced_features

    assert len(labels) == len(features)
    return features, labels


def preprocess(training_file='train_data.tsv', testing_file='devel_data.tsv'):

    # get training data from a file
    training_file_handler = codecs.open(training_file, 'r', encoding='utf-8')
    training_lines = training_file_handler.readlines()
    training_file_handler.close()

    # get testing data from a file
    testing_file_handler = codecs.open(testing_file, 'r', encoding='utf-8')
    testing_lines = testing_file_handler.readlines()
    testing_file_handler.close()

    # split data into features & labels
    features_train, labels_train = split_labels_features(training_lines)
    features_test, labels_test = split_labels_features(testing_lines)

    # stem
    stemmer = SnowballStemmer("english")
    features_train = [stem(stemmer, x) for x in features_train]
    features_test = [stem(stemmer, x) for x in features_test]

    # Convert features to matrix of token counts
    vectorizer = CountVectorizer(stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)

    # select best features
    selector = SelectPercentile(f_classif, percentile=5)
    selector.fit(features_train_transformed, labels_train)

    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed = selector.transform(features_test_transformed).toarray()

    print "# training data", len(labels_train)
    print "# testing data", len(labels_test)

    return features_train_transformed, features_test_transformed, labels_train, labels_test
