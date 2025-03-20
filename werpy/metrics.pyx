"""
This Cython module provides functions for calculating string matching metrics between 
reference and hypothesis strings. It contains two functions: calculations and metrics.
The calculations function takes two input sequences (reference and hypothesis) and 
returns a ragged array containing the word error rate (WER), Levenshtein distance (LD), 
number of words in the reference sequence, counts of insertions, deletions and 
substitutions, as well as lists of inserted, deleted and substituted words. The metrics 
function applies vectorization to the calculations function, enabling it to take in 
multiple values for reference and hypothesis in the form of lists or numpy arrays.

This Cython module provides efficient implementations of word error rate (WER) and 
Levenshtein distance (LD) calculations by utilizing C data types.

Functions:
- calculations(reference, hypothesis) -> np.ndarray: Calculates WER and related metrics 
for two input sequences and returns a ragged array containing the metrics.
- metrics(reference, hypothesis) -> np.ndarray: Applies vectorization to the 
calculations function to calculate WER and related metrics for multiple pairs of input 
sequences.
"""

import numpy as np
cimport numpy as cnp

cnp.import_array()

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray calculations(object reference, object hypothesis):
    cdef list reference_word = reference.split()
    cdef list hypothesis_word = hypothesis.split()

    cdef Py_ssize_t m, n, i, j, substitution_cost, ld, insertions, deletions, substitutions
    cdef list inserted_words, deleted_words, substituted_words
    m, n = len(reference_word), len(hypothesis_word)
    ldm = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                ldm[i][j] = j
            elif j == 0:
                ldm[i][j] = i
            else:
                substitution_cost = 0 if reference_word[i - 1] == hypothesis_word[j - 1] else 1
                ldm[i][j] = min(
                    ldm[i - 1][j] + 1,  # Deletion
                    ldm[i][j - 1] + 1,  # Insertion
                    ldm[i - 1][j - 1] + substitution_cost  # Substitution
                )

    ld = ldm[m][n]
    wer = ld / m

    insertions, deletions, substitutions = 0, 0, 0
    inserted_words, deleted_words, substituted_words = [], [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference_word[i - 1] == hypothesis_word[j - 1]:
            i -= 1
            j -= 1
        else:
            if i > 0 and j > 0 and ldm[i][j] == ldm[i - 1][j - 1] + 1:
                substitutions += 1
                substituted_words.append((reference_word[i - 1], hypothesis_word[j - 1]))
                i -= 1
                j -= 1
            elif j > 0 and ldm[i][j] == ldm[i][j - 1] + 1:
                insertions += 1
                inserted_words.append(hypothesis_word[j - 1])
                j -= 1
            elif i > 0 and ldm[i][j] == ldm[i - 1][j] + 1:
                deletions += 1
                deleted_words.append(reference_word[i - 1])
                i -= 1

    inserted_words.reverse(), deleted_words.reverse(), substituted_words.reverse()

    return np.array(
        [wer, ld, m, insertions, deletions, substitutions, inserted_words, deleted_words, substituted_words],
        dtype=object)

def metrics(reference, hypothesis):
    vectorize_calculations = np.vectorize(calculations)
    result = vectorize_calculations(reference, hypothesis)
    return result
