#!/usr/bin/env python3
"""
Function that calculates the unigram BLEU score for a sentence
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence
    """
    candidate = len(sentence)
    words = list(set(sentence))
    gram_dict = {}
    for ref in references:
        for gram in ref:
            if gram in words:
                if gram not in gram_dict:
                    gram_dict[gram] = ref.count(gram)
                else:
                    gram_dict[gram] = max(ref.count(gram), gram_dict[gram])
    prob = sum(gram_dict.values()) / candidate
    best = map(lambda ref: (abs(len(ref) - candidate), len(ref)), references)
    sorted_best = sorted(best, key=(lambda x: x[0]))
    best = sorted_best[0][1]
    bleu = 1 if candidate > best else np.exp(1 - (best / candidate))
    return bleu * np.exp(np.log(prob))
