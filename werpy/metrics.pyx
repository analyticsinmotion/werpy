# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
This Cython module provides functions for calculating string matching metrics between
reference and hypothesis strings. Word error rate (WER) and Levenshtein distance (LD) are
computed with C data types, and batches are processed in a C-level loop.

Three calculation paths are available, each with a single-pair function and a router that
accepts either a pair of strings or a pair of equal-length lists or numpy arrays:

- calculations(reference, hypothesis) -> np.ndarray: the full path. Returns a nine-element
  object array containing the WER, LD, number of words in the reference, counts of
  insertions, deletions and substitutions, and lists of the inserted, deleted and
  substituted words.
- metrics(reference, hypothesis) -> np.ndarray: routes to calculations for a pair of strings
  and returns a (9,) object array, or processes a batch and returns an (n, 9) object array
  with one row per pair.
- calculations_fast(reference, hypothesis) -> np.ndarray: the counts-only path. Returns a
  (6,) float64 array containing the WER, LD, number of words in the reference, and counts of
  insertions, deletions and substitutions. No word lists are built.
- metrics_fast(reference, hypothesis) -> np.ndarray: routes to calculations_fast for a pair
  of strings, or returns an (n, 6) float64 array for a batch.
- calculations_wer_only(reference, hypothesis) -> np.ndarray: the WER-only path. Returns a
  (3,) float64 array containing the WER, LD and number of words in the reference, computed
  with a two-row dynamic programming buffer.
- metrics_wer_only(reference, hypothesis) -> np.ndarray: routes to calculations_wer_only for
  a pair of strings, or returns an (n, 3) float64 array for a batch, reusing one pair of
  row buffers and one token buffer across the whole batch.

Before the edit distance is computed, every token of a pair is mapped to one canonical
object per distinct word, so that the dynamic programming loops compare words by pointer
equality rather than by string comparison.

Input validation is performed by werpy.errorhandler.error_handler before these functions
are called by the public werpy functions.
"""

import numpy as np
cimport numpy as cnp

cnp.import_array()

cimport cython
from cpython.ref cimport PyObject
from cpython.dict cimport PyDict_SetDefault
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _canonical_tokens(
    list reference_word,
    list hypothesis_word,
    PyObject** ref_tok,
    PyObject** hyp_tok,
) except -1:
    """
    Fill ref_tok and hyp_tok with one canonical object per distinct word, taken from the
    first occurrence of that word in either list. Two tokens are the same word exactly when
    their canonical pointers are equal. The pointers are borrowed from the token lists, which
    must stay alive while the buffers are in use.
    """
    cdef dict canonical = {}
    cdef Py_ssize_t k
    cdef object word

    for k in range(len(reference_word)):
        word = reference_word[k]
        ref_tok[k] = PyDict_SetDefault(canonical, word, word)
    for k in range(len(hypothesis_word)):
        word = hypothesis_word[k]
        hyp_tok[k] = PyDict_SetDefault(canonical, word, word)
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray calculations(object reference, object hypothesis):
    cdef list reference_word = reference.split()
    cdef list hypothesis_word

    # Use Py_ssize_t for indices and sizes
    # Py_ssize_t matches Python's internal index type and avoids unnecessary
    # casts or overflow risks when working with Python lists and memoryviews.
    cdef Py_ssize_t m = len(reference_word)
    cdef Py_ssize_t n
    cdef Py_ssize_t i, j

    # An identical pair has no edits, so the token mapping and the dynamic
    # programme are skipped. Only the reference word count is needed.
    if reference == hypothesis:
        return np.array([0.0, 0, m, 0, 0, 0, [], [], []], dtype=object)

    hypothesis_word = hypothesis.split()
    n = len(hypothesis_word)

    # Metrics and outputs
    cdef int ld, insertions, deletions, substitutions
    cdef double wer
    cdef list inserted_words, deleted_words, substituted_words

    # Variables for optimized DP loop
    cdef int cost, del_cost, ins_cost, sub_cost, best

    # Canonical token pointers, reference tokens first, hypothesis tokens after them
    cdef PyObject** ref_tok
    cdef PyObject** hyp_tok

    # Initialize the Levenshtein distance matrix
    # Use empty instead of zeros to avoid redundant initialization.
    # SAFETY: All cells are explicitly initialized below (row 0, col 0, then DP loop).
    # Allocate the (m+1) x (n+1) DP matrix without zero-initialization to avoid
    # redundant memory writes. Boundary conditions are initialized explicitly.
    cdef cnp.int32_t[:, :] ldm = np.empty((m + 1, n + 1), dtype=np.int32)

    # Initialize first column and first row (boundary conditions)
    for i in range(m + 1):
        ldm[i, 0] = <cnp.int32_t>i
    for j in range(n + 1):
        ldm[0, j] = <cnp.int32_t>j

    ref_tok = <PyObject**>PyMem_Malloc((m + n) * sizeof(PyObject*))
    if ref_tok == NULL:
        raise MemoryError()
    hyp_tok = ref_tok + m

    try:
        _canonical_tokens(reference_word, hypothesis_word, ref_tok, hyp_tok)

        # Fill the Levenshtein distance matrix
        # Compute edit distances using a branch-free inner loop and manual minimum
        # selection to keep all operations at C level and minimize per-cell overhead.
        # No boundary condition branches in the hot loop, manual min selection.
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else 1

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
            if i > 0 and j > 0 and ref_tok[i - 1] == hyp_tok[j - 1]:
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
    finally:
        PyMem_Free(ref_tok)

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
    cdef list hypothesis_word

    cdef Py_ssize_t m = len(reference_word)
    cdef Py_ssize_t n
    cdef Py_ssize_t i, j

    # An identical pair has no edits, so the token mapping and the dynamic
    # programme are skipped. Only the reference word count is needed.
    if reference == hypothesis:
        return np.array([0.0, 0.0, <double>m, 0.0, 0.0, 0.0], dtype=np.float64)

    hypothesis_word = hypothesis.split()
    n = len(hypothesis_word)

    cdef int ld, insertions, deletions, substitutions
    cdef double wer

    cdef int cost, del_cost, ins_cost, sub_cost, best

    # Canonical token pointers, reference tokens first, hypothesis tokens after them
    cdef PyObject** ref_tok
    cdef PyObject** hyp_tok

    # Allocate the (m+1) x (n+1) DP matrix without zero-initialization
    cdef cnp.int32_t[:, :] ldm = np.empty((m + 1, n + 1), dtype=np.int32)

    # Initialize first column and first row (boundary conditions)
    for i in range(m + 1):
        ldm[i, 0] = <cnp.int32_t>i
    for j in range(n + 1):
        ldm[0, j] = <cnp.int32_t>j

    ref_tok = <PyObject**>PyMem_Malloc((m + n) * sizeof(PyObject*))
    if ref_tok == NULL:
        raise MemoryError()
    hyp_tok = ref_tok + m

    try:
        _canonical_tokens(reference_word, hypothesis_word, ref_tok, hyp_tok)

        # Fill the Levenshtein distance matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else 1

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
            if i > 0 and j > 0 and ref_tok[i - 1] == hyp_tok[j - 1]:
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
    finally:
        PyMem_Free(ref_tok)

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
    cdef list hypothesis_word

    cdef Py_ssize_t m = len(reference_word)
    cdef Py_ssize_t n

    cdef Py_ssize_t i, j
    cdef int cost, del_cost, ins_cost, sub_cost, best, ld
    cdef double wer

    # Canonical token pointers, reference tokens first, hypothesis tokens after them
    cdef PyObject** ref_tok
    cdef PyObject** hyp_tok

    cdef cnp.ndarray prev_arr
    cdef cnp.ndarray curr_arr
    cdef cnp.int32_t[:] prev
    cdef cnp.int32_t[:] curr

    # An identical pair has no edits, so the token mapping and the dynamic
    # programme are skipped. Only the reference word count is needed.
    if reference == hypothesis:
        return np.array([0.0, 0.0, <double>m], dtype=np.float64)

    hypothesis_word = hypothesis.split()
    n = len(hypothesis_word)

    prev_arr = np.empty(n + 1, dtype=np.int32)
    curr_arr = np.empty(n + 1, dtype=np.int32)
    prev = prev_arr
    curr = curr_arr

    for j in range(n + 1):
        prev[j] = <cnp.int32_t>j

    ref_tok = <PyObject**>PyMem_Malloc((m + n) * sizeof(PyObject*))
    if ref_tok == NULL:
        raise MemoryError()
    hyp_tok = ref_tok + m

    try:
        _canonical_tokens(reference_word, hypothesis_word, ref_tok, hyp_tok)

        for i in range(1, m + 1):
            curr[0] = <cnp.int32_t>i
            for j in range(1, n + 1):
                cost = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else 1

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
    finally:
        PyMem_Free(ref_tok)

    ld = prev[n]
    wer = (<double>ld) / m if m > 0 else 0.0

    return np.array([wer, <double>ld, <double>m], dtype=np.float64)


cdef struct _BatchBuffers:
    # Two dynamic programming rows and one token buffer shared across a batch.
    # Each is grown on demand and freed once when the batch is finished.
    cnp.int32_t* prev
    cnp.int32_t* curr
    Py_ssize_t row_capacity
    PyObject** tok
    Py_ssize_t tok_capacity


cdef inline int _ensure_batch_buffers(
    _BatchBuffers* buf,
    Py_ssize_t rows,
    Py_ssize_t tokens,
) except -1:
    """
    Make each dynamic programming row hold at least `rows` cells and the token buffer
    hold at least `tokens` pointers. Capacity at least doubles on every growth, so a
    batch performs a bounded number of allocations however its lengths are ordered.
    """
    cdef Py_ssize_t capacity
    cdef cnp.int32_t* row_mem
    cdef PyObject** tok_mem

    if rows > buf.row_capacity:
        capacity = 2 * buf.row_capacity
        if capacity < rows:
            capacity = rows
        row_mem = <cnp.int32_t*>PyMem_Realloc(buf.prev, 2 * capacity * sizeof(cnp.int32_t))
        if row_mem == NULL:
            raise MemoryError()
        buf.prev = row_mem
        buf.curr = row_mem + capacity
        buf.row_capacity = capacity

    if tokens > buf.tok_capacity:
        capacity = 2 * buf.tok_capacity
        if capacity < tokens:
            capacity = tokens
        tok_mem = <PyObject**>PyMem_Realloc(buf.tok, capacity * sizeof(PyObject*))
        if tok_mem == NULL:
            raise MemoryError()
        buf.tok = tok_mem
        buf.tok_capacity = capacity

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _calculations_wer_only_reuse_ptr(
    object reference,
    object hypothesis,
    _BatchBuffers* buf,
    double* out3,
) except -1:
    """
    Internal WER-only DP using caller-provided buffers and pointer swap (no copying).
    Writes: out3[0]=wer, out3[1]=ld, out3[2]=m

    This implementation uses true pointer swapping instead of copying values,
    eliminating O(n) copy overhead per outer iteration. The rows and the token
    buffer are owned by the caller and grown here when a pair needs more room.
    """
    cdef list reference_word = reference.split()
    cdef list hypothesis_word

    cdef Py_ssize_t m = len(reference_word)
    cdef Py_ssize_t n

    cdef Py_ssize_t i, j
    cdef int cost, del_cost, ins_cost, sub_cost, best, ld
    cdef cnp.int32_t* prev
    cdef cnp.int32_t* curr
    cdef cnp.int32_t* tmp

    # Canonical token pointers, reference tokens first, hypothesis tokens after them
    cdef PyObject** ref_tok
    cdef PyObject** hyp_tok

    # An identical pair has no edits, so the token mapping and the dynamic
    # programme are skipped. Only the reference word count is needed.
    if reference == hypothesis:
        out3[0] = 0.0
        out3[1] = 0.0
        out3[2] = <double>m
        return 0

    hypothesis_word = hypothesis.split()
    n = len(hypothesis_word)

    _ensure_batch_buffers(buf, n + 1, m + n)
    prev = buf.prev
    curr = buf.curr
    ref_tok = buf.tok
    hyp_tok = ref_tok + m

    # Initialize base row: prev[j] = j for j=0..n
    for j in range(n + 1):
        prev[j] = <cnp.int32_t>j

    _canonical_tokens(reference_word, hypothesis_word, ref_tok, hyp_tok)

    for i in range(1, m + 1):
        curr[0] = <cnp.int32_t>i
        for j in range(1, n + 1):
            cost = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else 1

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
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray _metrics_batch_wer_only(list references, list hypotheses):
    """
    Fast batch processing for WER-only calculations with buffer reuse and pointer swapping.

    One pair of dynamic programming rows and one token buffer are shared by every pair in
    the batch. They are grown on demand as longer pairs are met and freed once at the end,
    so the hypotheses are split only inside the loop. Uses true pointer swapping instead
    of value copying for optimal performance.

    Returns (n, 3) float64 array where each row contains:
    [wer, ld, m]
    """
    cdef Py_ssize_t n_pairs = len(references)
    cdef Py_ssize_t idx

    cdef cnp.ndarray out = np.empty((n_pairs, 3), dtype=np.float64)
    cdef double* out_data = <double*>cnp.PyArray_DATA(out)

    cdef _BatchBuffers buf
    buf.prev = NULL
    buf.curr = NULL
    buf.row_capacity = 0
    buf.tok = NULL
    buf.tok_capacity = 0

    # Process each pair using the shared buffers, writing directly to output rows
    try:
        for idx in range(n_pairs):
            _calculations_wer_only_reuse_ptr(references[idx], hypotheses[idx], &buf, out_data + idx * 3)
    finally:
        PyMem_Free(buf.prev)
        PyMem_Free(buf.tok)

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
