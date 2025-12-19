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
    wer = (<double>ld) / m if m > 0 else 0.0

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
    cdef Py_ssize_t idx

    # Rows output, dtype=object because cols 6-8 are lists
    cdef cnp.ndarray out = np.empty((n, 9), dtype=object)

    cdef object r
    for idx in range(n):
        r = calculations(references[idx], hypotheses[idx])

        # Safety: unwrap 0-D wrappers if they ever occur
        if isinstance(r, np.ndarray) and r.ndim == 0:
            r = r.item()

        out[idx, :] = r

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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray calculations_fast(object reference, object hypothesis):
    """
    Fast path for WER/LD calculations without word tracking.
    Returns only numeric metrics (WER, LD, m, insertions, deletions, substitutions).

    This function is optimized for use cases that only need counts and metrics,
    not the actual lists of inserted/deleted/substituted words.

    Returns (6,) float64 array: [wer, ld, m, insertions, deletions, substitutions]
    """
    cdef list reference_word = reference.split()
    cdef list hypothesis_word = hypothesis.split()

    cdef Py_ssize_t m = len(reference_word)
    cdef Py_ssize_t n = len(hypothesis_word)
    cdef Py_ssize_t i, j

    cdef int ld, insertions, deletions, substitutions
    cdef double wer

    cdef int cost, del_cost, ins_cost, sub_cost, best

    # Allocate the (m+1) x (n+1) DP matrix without zero-initialization
    cdef int[:, :] ldm = np.empty((m + 1, n + 1), dtype=np.int32)

    # Initialize first column and first row (boundary conditions)
    for i in range(m + 1):
        ldm[i, 0] = <int>i
    for j in range(n + 1):
        ldm[0, j] = <int>j

    # Fill the Levenshtein distance matrix
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
    wer = (<double>ld) / m if m > 0 else 0.0

    # Backtrace to count errors (no word tracking)
    insertions, deletions, substitutions = 0, 0, 0
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference_word[i - 1] == hypothesis_word[j - 1]:
            i -= 1
            j -= 1
        else:
            if i > 0 and j > 0 and ldm[i, j] == ldm[i - 1, j - 1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif j > 0 and ldm[i, j] == ldm[i, j - 1] + 1:
                insertions += 1
                j -= 1
            elif i > 0 and ldm[i, j] == ldm[i - 1, j] + 1:
                deletions += 1
                i -= 1

    return np.array(
        [wer, <double>ld, <double>m,
         <double>insertions, <double>deletions, <double>substitutions],
        dtype=np.float64
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray _metrics_batch_fast(list references, list hypotheses):
    """
    Fast batch processing without word tracking.

    Returns (n, 6) float64 array where each row contains:
    [wer, ld, m, insertions, deletions, substitutions]
    """
    cdef Py_ssize_t n = len(references)
    cdef Py_ssize_t idx

    cdef cnp.ndarray out = np.empty((n, 6), dtype=np.float64)

    cdef cnp.ndarray r
    for idx in range(n):
        r = calculations_fast(references[idx], hypotheses[idx])
        out[idx, :] = r

    return out


cpdef object metrics_fast(object reference, object hypothesis):
    """
    Fast metrics entry point without word tracking.

    Returns:
    - strings: (6,) float64 array [wer, ld, m, insertions, deletions, substitutions]
    - sequences: (n, 6) float64 array, one row per pair
    """
    if isinstance(reference, (list, np.ndarray)) and isinstance(hypothesis, (list, np.ndarray)):
        return _metrics_batch_fast(list(reference), list(hypothesis))
    return calculations_fast(reference, hypothesis)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray calculations_wer_only(object reference, object hypothesis):
    """
    WER-only fast path - 2-row DP (O(n) memory), no backtrace.
    Returns only [wer, ld, m] without error counts or word tracking.

    This is the fastest path for pure WER calculation, using space-optimized
    Wagner-Fischer algorithm with rolling 2-row buffer instead of full matrix.

    Returns (3,) float64 array: [wer, ld, m]
    """
    cdef list reference_word = reference.split()
    cdef list hypothesis_word = hypothesis.split()

    cdef Py_ssize_t m = len(reference_word)
    cdef Py_ssize_t n = len(hypothesis_word)

    cdef Py_ssize_t i, j
    cdef int cost, del_cost, ins_cost, sub_cost, best, ld
    cdef double wer

    cdef cnp.ndarray prev_arr = np.empty(n + 1, dtype=np.int32)
    cdef cnp.ndarray curr_arr = np.empty(n + 1, dtype=np.int32)

    cdef int[:] prev = prev_arr
    cdef int[:] curr = curr_arr

    for j in range(n + 1):
        prev[j] = <int>j

    for i in range(1, m + 1):
        curr[0] = <int>i
        for j in range(1, n + 1):
            cost = 0 if reference_word[i - 1] == hypothesis_word[j - 1] else 1

            del_cost = prev[j] + 1
            ins_cost = curr[j - 1] + 1
            sub_cost = prev[j - 1] + cost

            best = del_cost
            if ins_cost < best:
                best = ins_cost
            if sub_cost < best:
                best = sub_cost

            curr[j] = best

        prev, curr = curr, prev

    ld = prev[n]
    wer = (<double>ld) / m if m > 0 else 0.0

    return np.array([wer, <double>ld, <double>m], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _calculations_wer_only_reuse_ptr(
    object reference,
    object hypothesis,
    cnp.int32_t* prev,
    cnp.int32_t* curr,
    double* out3,
) except *:
    """
    Internal WER-only DP using caller-provided buffers and pointer swap (no copying).
    Writes: out3[0]=wer, out3[1]=ld, out3[2]=m

    This implementation uses true pointer swapping instead of copying values,
    eliminating O(n) copy overhead per outer iteration.
    """
    cdef list reference_word = reference.split()
    cdef list hypothesis_word = hypothesis.split()

    cdef Py_ssize_t m = len(reference_word)
    cdef Py_ssize_t n = len(hypothesis_word)

    cdef Py_ssize_t i, j
    cdef int cost, del_cost, ins_cost, sub_cost, best, ld
    cdef cnp.int32_t* tmp

    # Initialize base row: prev[j] = j for j=0..n
    for j in range(n + 1):
        prev[j] = j

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if reference_word[i - 1] == hypothesis_word[j - 1] else 1

            del_cost = prev[j] + 1
            ins_cost = curr[j - 1] + 1
            sub_cost = prev[j - 1] + cost

            best = del_cost
            if ins_cost < best:
                best = ins_cost
            if sub_cost < best:
                best = sub_cost

            curr[j] = best

        # Swap prev and curr pointers (zero-cost operation)
        tmp = prev
        prev = curr
        curr = tmp

    ld = prev[n]
    out3[0] = (<double>ld) / m if m > 0 else 0.0
    out3[1] = <double>ld
    out3[2] = <double>m


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray _metrics_batch_wer_only(list references, list hypotheses):
    """
    Fast batch processing for WER-only calculations with buffer reuse and pointer swapping.

    Eliminates repeated buffer allocations by reusing prev/curr arrays across all pairs
    in the batch, sized to the maximum hypothesis length. Uses true pointer swapping
    instead of value copying for optimal performance.

    Returns (n, 3) float64 array where each row contains:
    [wer, ld, m]
    """
    cdef Py_ssize_t n_pairs = len(references)
    cdef Py_ssize_t idx

    cdef cnp.ndarray out = np.empty((n_pairs, 3), dtype=np.float64)

    # Find max hypothesis token length to size buffers once
    cdef Py_ssize_t max_n = 0
    cdef Py_ssize_t this_n
    cdef object h
    cdef list h_words
    for idx in range(n_pairs):
        h = hypotheses[idx]
        h_words = h.split()
        this_n = len(h_words)
        if this_n > max_n:
            max_n = this_n

    # Allocate reusable DP buffers once for the entire batch
    cdef cnp.ndarray prev_arr = np.empty(max_n + 1, dtype=np.int32)
    cdef cnp.ndarray curr_arr = np.empty(max_n + 1, dtype=np.int32)

    # Get raw pointers for zero-cost swapping
    cdef cnp.int32_t* prev = <cnp.int32_t*>cnp.PyArray_DATA(prev_arr)
    cdef cnp.int32_t* curr = <cnp.int32_t*>cnp.PyArray_DATA(curr_arr)

    # Process each pair using shared buffers, writing directly to output rows
    cdef double* out_row
    for idx in range(n_pairs):
        out_row = <double*>cnp.PyArray_DATA(out) + (idx * 3)
        _calculations_wer_only_reuse_ptr(references[idx], hypotheses[idx], prev, curr, out_row)

    return out


cpdef object metrics_wer_only(object reference, object hypothesis):
    """
    WER-only metrics entry point (fastest path).

    Returns:
    - strings: (3,) float64 array [wer, ld, m]
    - sequences: (n, 3) float64 array, one row per pair
    """
    if isinstance(reference, (list, np.ndarray)) and isinstance(hypothesis, (list, np.ndarray)):
        return _metrics_batch_wer_only(list(reference), list(hypothesis))
    return calculations_wer_only(reference, hypothesis)
