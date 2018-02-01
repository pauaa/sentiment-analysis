# -*- coding: utf-8 -*-
from collections import defaultdict
from preprocess import POSITIVE, NEGATIVE


def fscore(true_labels, predicted_labels):
    """
    Given true and predicted labels, calculate the macro-averaged f-score.
    :param true_labels: List of actual labels
    :param predicted_labels: List of classified labels
    :return: F-score
    """
    assert len(true_labels) == len(predicted_labels)
    counts = defaultdict(lambda: defaultdict(int))

    counts[POSITIVE]['all_true'] = list(true_labels).count(POSITIVE)
    counts[NEGATIVE]['all_true'] = list(true_labels).count(NEGATIVE)

    for i, true_label in enumerate(true_labels):
        predicted = predicted_labels[i]
        if predicted == true_label and predicted in {POSITIVE, NEGATIVE}:
            counts[predicted]['true_positive'] += 1
        elif predicted != true_label and predicted in {POSITIVE, NEGATIVE}:
            counts[predicted]['false_positive'] += 1

    try:
        precision_positive = float(counts[POSITIVE]['true_positive']) / (
                counts[POSITIVE]['true_positive'] + counts[POSITIVE]['false_positive'])
        recall_positive = float(counts[POSITIVE]['true_positive']) / counts[POSITIVE]['all_true']
        f_score_positive = 2 * precision_positive * recall_positive / (
                precision_positive + recall_positive)
    except:
        f_score_positive = 0.0

    try:
        precision_negative = float(counts[NEGATIVE]['true_positive']) / (
                counts[NEGATIVE]['true_positive'] + counts[NEGATIVE]['false_positive'])
        recall_negative = float(counts[NEGATIVE]['true_positive']) / counts[NEGATIVE]['all_true']
        f_score_negative = 2 * precision_negative * recall_negative / (
                precision_negative + recall_negative)
    except:
        f_score_negative = 0.0

    return (f_score_positive + f_score_negative) / 2.0
