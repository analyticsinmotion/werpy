# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

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
    cdef int insert_idx = 0, delete_idx = 0, substitute_idx = 0
    cdef int max_len = max(len(reference_word), len(hypothesis_word))
    cdef list inserted_words = [None] * max_len
    cdef list deleted_words = [None] * max_len
    cdef list substituted_words = [None] * max_len
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference_word[i - 1] == hypothesis_word[j - 1]:
            i -= 1
            j -= 1
        else:
            if i > 0 and j > 0 and ldm[i][j] == ldm[i - 1][j - 1] + 1:
                substitutions += 1
                # Insert at the current substitute_idx to preserve reverse order
                substituted_words[substitute_idx] = (reference_word[i - 1], hypothesis_word[j - 1])
                substitute_idx += 1
                i -= 1
                j -= 1
            elif j > 0 and ldm[i][j] == ldm[i][j - 1] + 1:
                insertions += 1
                # Insert at the current insert_idx to preserve reverse order
                inserted_words[insert_idx] = hypothesis_word[j - 1]
                insert_idx += 1
                j -= 1
            elif i > 0 and ldm[i][j] == ldm[i - 1][j] + 1:
                deletions += 1
                # Insert at the current delete_idx to preserve reverse order
                deleted_words[delete_idx] = reference_word[i - 1]
                delete_idx += 1
                i -= 1

    # Slice off trailing None values
    inserted_words = inserted_words[:insert_idx]
    deleted_words = deleted_words[:delete_idx]
    substituted_words = substituted_words[:substitute_idx]

    return np.array(
        [wer, ld, m, insertions, deletions, substitutions, inserted_words, deleted_words, substituted_words],
        dtype=object
    )

def metrics(reference, hypothesis):
    vectorize_calculations = np.vectorize(calculations)
    result = vectorize_calculations(reference, hypothesis)
    return result
