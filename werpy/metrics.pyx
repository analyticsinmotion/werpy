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

    # Use Py_ssize_t for indices and sizes
    # Py_ssize_t matches Python's internal index type and avoids unnecessary
    # casts or overflow risks when working with Python lists and memoryviews.
    cdef Py_ssize_t m = len(reference_word)
    cdef Py_ssize_t n = len(hypothesis_word)
    cdef Py_ssize_t i, j

    # Metrics and outputs
    cdef int ld, insertions, deletions, substitutions
    cdef double wer
    cdef list inserted_words, deleted_words, substituted_words

    # Variables for optimized DP loop
    cdef int cost, del_cost, ins_cost, sub_cost, best

    # Initialize the Levenshtein distance matrix
    # Use empty instead of zeros to avoid redundant initialization.
    # SAFETY: All cells are explicitly initialized below (row 0, col 0, then DP loop).
    # Allocate the (m+1) x (n+1) DP matrix without zero-initialization to avoid
    # redundant memory writes. Boundary conditions are initialized explicitly.
    cdef int[:, :] ldm = np.empty((m + 1, n + 1), dtype=np.int32)

    # Initialize first column and first row (boundary conditions)
    for i in range(m + 1):
        ldm[i, 0] = <int>i
    for j in range(n + 1):
        ldm[0, j] = <int>j

    # Fill the Levenshtein distance matrix
    # Compute edit distances using a branch-free inner loop and manual minimum
    # selection to keep all operations at C level and minimize per-cell overhead.
    # No boundary condition branches in the hot loop, manual min selection.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if reference_word[i - 1] == hypothesis_word[j - 1] else 1

            del_cost = ldm[i - 1, j] + 1
            ins_cost = ldm[i, j - 1] + 1
            sub_cost = ldm[i - 1, j - 1] + cost

            best = del_cost
            if ins_cost < best:
                best = ins_cost
            if sub_cost < best:
                best = sub_cost

            ldm[i, j] = best

    ld = ldm[m, n]
    wer = (<double>ld) / m

    insertions, deletions, substitutions = 0, 0, 0
    inserted_words, deleted_words, substituted_words = [], [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference_word[i - 1] == hypothesis_word[j - 1]:
            i -= 1
            j -= 1
        else:
            if i > 0 and j > 0 and ldm[i, j] == ldm[i - 1, j - 1] + 1:
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

    inserted_words.reverse(), deleted_words.reverse(), substituted_words.reverse()

    return np.array(
        [wer, ld, m, insertions, deletions, substitutions, inserted_words, deleted_words, substituted_words],
        dtype=object)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray _metrics_batch(list references, list hypotheses):
    """
    Private batch processing function. Processes multiple reference-hypothesis
    pairs at C speed, eliminating np.vectorize overhead.

    Returns (n, 9) object array where each row contains:
    [wer, ld, m, insertions, deletions, substitutions, inserted_words, deleted_words, substituted_words]
    """
    cdef Py_ssize_t n = len(references)
    cdef Py_ssize_t idx, j

    # Rows output, dtype=object because cols 6-8 are lists
    cdef cnp.ndarray out = np.empty((n, 9), dtype=object)

    cdef object r
    for idx in range(n):
        r = calculations(references[idx], hypotheses[idx])

        # Safety: unwrap 0-D wrappers if they ever occur
        if isinstance(r, np.ndarray) and r.ndim == 0:
            r = r.item()

        for j in range(9):
            out[idx, j] = r[j]

    return out


cpdef object metrics(object reference, object hypothesis):
    """
    Unified fast metrics entry point (Option A, rows contract).

    Returns:
    - strings: a single row (len 9)
    - sequences: an (n, 9) object ndarray, one row per pair
    """
    if isinstance(reference, (list, np.ndarray)) and isinstance(hypothesis, (list, np.ndarray)):
        return _metrics_batch(list(reference), list(hypothesis))
    return calculations(reference, hypothesis)
