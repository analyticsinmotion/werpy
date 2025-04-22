# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
This Cython module provides efficient implementations of word error rate (WER) and 
Levenshtein distance (LD) calculations using native C types and optimized loops.

Functions:
- calculations(reference, hypothesis) -> Result: Calculates WER and related metrics for two sequences.
- metrics(reference_list, hypothesis_list) -> list[Result]: Applies the calculations to multiple pairs.
"""

import numpy as np
cimport numpy as cnp

cimport cython

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Result:
    cdef public double wer
    cdef public int ld, m, insertions, deletions, substitutions
    cdef public list inserted_words, deleted_words, substituted_words

    def to_dict(self):
        return {
            "wer": self.wer,
            "ld": self.ld,
            "m": self.m,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "substitutions": self.substitutions,
            "inserted_words": self.inserted_words,
            "deleted_words": self.deleted_words,
            "substituted_words": self.substituted_words
        }

cpdef Result calculations(object reference, object hypothesis):
    cdef list reference_word = reference.split()
    cdef list hypothesis_word = hypothesis.split()

    cdef Py_ssize_t m = len(reference_word)
    cdef Py_ssize_t n = len(hypothesis_word)
    cdef Py_ssize_t i, j
    cdef int substitution_cost, ld, insertions, deletions, substitutions
    cdef list inserted_words, deleted_words, substituted_words

    # Initialize the Levenshtein distance matrix
    cdef int[:, :] ldm = np.zeros((m + 1, n + 1), dtype=np.int32)

    # Fill the Levenshtein distance matrix
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                ldm[i, j] = j
            elif j == 0:
                ldm[i, j] = i
            else:
                substitution_cost = 0 if reference_word[i - 1] == hypothesis_word[j - 1] else 1
                ldm[i, j] = min(
                    ldm[i - 1, j] + 1, # Deletion
                    ldm[i, j - 1] + 1, # Insertion
                    ldm[i - 1, j - 1] + substitution_cost # Substitution
                )

    ld = ldm[m, n]
    wer = ld / m

    insertions = deletions = substitutions = 0
    inserted_words, deleted_words, substituted_words = [], [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference_word[i - 1] == hypothesis_word[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and ldm[i, j] == ldm[i - 1, j - 1] + 1:
            substitutions += 1
            substituted_words.append((reference_word[i - 1], hypothesis_word[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and ldm[i, j] == ldm[i, j - 1] + 1:
            insertions += 1
            inserted_words.append(hypothesis_word[j - 1])
            j -= 1
        elif i > 0 and ldm[i, j] == ldm[i - 1, j] + 1:
            deletions += 1
            deleted_words.append(reference_word[i - 1])
            i -= 1

    inserted_words.reverse()
    deleted_words.reverse()
    substituted_words.reverse()

cpdef Result calculations(object reference, object hypothesis):
    cdef Result result = Result()
    result.wer = wer
    result.ld = ld
    result.m = m
    result.insertions = insertions
    result.deletions = deletions
    result.substitutions = substitutions
    result.inserted_words = inserted_words
    result.deleted_words = deleted_words
    result.substituted_words = substituted_words

    return result


cpdef list metrics(object reference_list, object hypothesis_list):
    """
    Apply the calculations function to each pair of reference and hypothesis.
    Returns a list of Result objects.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t size = len(reference_list)
    cdef list results = []
    for i in range(size):
        results.append(calculations(reference_list[i], hypothesis_list[i]))
    return results

