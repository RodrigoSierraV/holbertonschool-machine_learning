#!/usr/bin/env python3
"""F1 score of a Confusion matrix """
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score of a confusion matrix
        f1_score = 2 * precision * sensitivity/(precision + sensitivity)
    """
    prec = precision(confusion)
    sens = sensitivity(confusion)
    f1 = 2 * prec * sens / (prec + sens)
    return f1
