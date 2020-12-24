#!/usr/bin/env python3
"""
Function that calculates the n-gram BLEU score for a sentence
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    :param: references is a list of ref translations
        each ref translation is a list of the words in the translation
    :param: sentence is a list containing the model proposed sentence
    :param: n is the size of the n-gram to use for evaluation
    Returns: the n-gram BLEU score
    """
    sen_len = len(sentence)
    calc_precision = precision(references, sentence, n)
    ref_idx = np.argmin([abs(len(x) - sen_len) for x in references])
    ref_len = len(references[ref_idx])
    bleu = 1 if sen_len > ref_len else np.exp(1 - (ref_len / sen_len))
    return bleu * calc_precision


def precision(references, sentence, n):
    """
    Calculates precision
    """
    references, sentence = to_bigrams(references, sentence, n)
    ref_dict = {}
    for ref in references:
        for gram in ref:
            if gram not in ref_dict:
                ref_dict[gram] = ref.count(gram)
            else:
                ref_dict[gram] = max(ref.count(gram), ref_dict[gram])
    gram_count = {}
    for ref in references:
        for gram in sentence:
            if gram in ref:
                gram_count[gram] = sentence.count(gram)

    for gram in gram_count:
        if gram in ref_dict:
            gram_count[gram] = min(ref_dict[gram], gram_count[gram])

    return sum(gram_count.values()) / len(sentence)


def to_bigrams(references, sentence, n):
    """
    words lists to bigrams
    """
    if n == 1:
        return references, sentence
    new_sentence = bigrams(sentence, n)
    new_ref = [bigrams(ref, n) for ref in references]
    return new_ref, new_sentence


def bigrams(sentence, n):
    """
    creates bigrams
    """
    bi_grams = []
    for i in range(len(sentence) - 1):
        bi_grams.append(' '.join(sentence[i:i+2]))
    return bi_grams
