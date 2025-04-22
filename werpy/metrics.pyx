# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
This Cython module provides efficient implementations of word error rate (WER) and 
Levenshtein distance (LD) calculations using native C types and optimized loops.

Functions:
- calculations(reference, hypothesis) -> Result: Dummy version for timing.
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
    cdef Result result = Result()
    result.wer = 0.1
    result.ld = 1
    result.m = 4
    result.insertions = 0
    result.deletions = 0
    result.substitutions = 0
    # Skip creating lists:
    result.inserted_words = None
    result.deleted_words = None
    result.substituted_words = None
    return result

cpdef list metrics(object reference_list, object hypothesis_list):
    """
    Apply the dummy calculations function to each pair of reference and hypothesis.
    Returns a list of Result objects.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t size = len(reference_list)
    cdef list results = []
    for i in range(size):
        results.append(calculations(reference_list[i], hypothesis_list[i]))
    return results
