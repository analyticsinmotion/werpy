# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
This Cython module provides the alignments of the weighted functions werp, werps and summaryp
for weights that decide between the alignments with the fewest edits.

Among the alignments with the fewest edits, the weighted cost changes with the number of
substitutions only. The lowest weighted cost belongs to the alignment with the fewest
substitutions when a substitution weighs more than the mean of an insertion and a deletion
(order 1), and to the alignment with the most substitutions when it weighs less (order -1).
werpy._errorhandler.substitution_order gives the order. For order 0 all these alignments have
the same weighted cost, and the weighted functions use werpy._metrics.

Each cell of the dynamic programme holds one integer key: (min(m, n) + 1) times the number of
edits, plus the number of substitutions for order 1 or minus it for order -1. The keys order the
alignments by their edits first and by their substitutions second.

- calculations_weighted(reference, hypothesis, order) -> np.ndarray: the full path. Returns the
  nine-element object array of werpy._metrics.calculations for that alignment: the WER, LD,
  number of words in the reference, counts of insertions, deletions and substitutions, and
  lists of the inserted, deleted and substituted words.
- metrics_weighted(reference, hypothesis, order) -> np.ndarray: routes to calculations_weighted
  for a pair of strings and returns a (9,) object array, or processes a batch and returns an
  (n, 9) object array with one row per pair.
- calculations_fast_weighted(reference, hypothesis, order) -> np.ndarray: the counts-only path.
  Returns the (6,) float64 array of werpy._metrics.calculations_fast for that alignment: the
  WER, LD, number of words in the reference, and counts of insertions, deletions and
  substitutions. It keeps two rows of keys and has no backtrace.
- metrics_fast_weighted(reference, hypothesis, order) -> np.ndarray: routes to
  calculations_fast_weighted for a pair of strings, or returns an (n, 6) float64 array for a
  batch.

Each function raises a ValueError when the order is not 1 or -1.

Input validation is performed by werpy._errorhandler.error_handler before these functions
are called by the public werpy functions.
"""

import numpy as np

cimport cython
from cpython.ref cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport int64_t, SIZE_MAX

cdef extern from "_dictcompat.h":
    PyObject* werpy_dict_setdefault(object p, object key, object default) except NULL


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
        ref_tok[k] = werpy_dict_setdefault(canonical, word, word)
    for k in range(len(hypothesis_word)):
        word = hypothesis_word[k]
        hyp_tok[k] = werpy_dict_setdefault(canonical, word, word)
    return 0


cdef inline int _check_equal_lengths(list references, list hypotheses) except -1:
    """
    Raise a ValueError when references and hypotheses have different numbers of elements.
    """
    if len(references) != len(hypotheses):
        raise ValueError("The Reference and Hypothesis input parameters must have the same number of elements.")
    return 0


cdef inline int _check_order(int order) except -1:
    """
    Raise a ValueError unless the order is 1 or -1.
    """
    if order != 1 and order != -1:
        raise ValueError("The order must be 1 or -1.")
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object calculations_weighted(object reference, object hypothesis, int order):
    """
    Full path for weights that decide between the alignments with the fewest edits.

    Among those alignments, it takes the one with the fewest substitutions when order is 1 and
    the one with the most substitutions when order is -1. The backtrace reads the texts from the
    end and takes, at each cell, the first move that keeps the key optimal: a match or a
    substitution, then an insertion, then a deletion, as in werpy._metrics.calculations.

    Returns (9,) object array: [wer, ld, m, insertions, deletions, substitutions,
    inserted_words, deleted_words, substituted_words]
    """
    _check_order(order)

    cdef list reference_word = reference.split()
    cdef list hypothesis_word

    cdef Py_ssize_t m = len(reference_word)
    cdef Py_ssize_t n, width
    cdef Py_ssize_t i, j

    # An identical pair has no edits, so the token mapping and the dynamic
    # programme are skipped. Only the reference word count is needed.
    if reference == hypothesis:
        return np.array([0.0, 0, m, 0, 0, 0, [], [], []], dtype=object)

    hypothesis_word = hypothesis.split()
    n = len(hypothesis_word)
    width = n + 1

    # Metrics and outputs
    cdef int ld, insertions, deletions, substitutions
    cdef double wer
    cdef list inserted_words, deleted_words, substituted_words
    cdef bint same

    # Variables of the dynamic programme. edit_step is larger than any number of
    # substitutions, so a key orders alignments by their edits first.
    cdef int64_t edit_step, sub_step, step, best, left, diag, up
    cdef int64_t* keys

    # Canonical token pointers, reference tokens first, hypothesis tokens after them
    cdef PyObject** ref_tok
    cdef PyObject** hyp_tok

    edit_step = (m if m < n else n) + 1
    sub_step = edit_step + order

    # The (m+1) x (n+1) key matrix, row by row. A size that cannot be addressed raises MemoryError.
    if <size_t>(m + 1) > SIZE_MAX // sizeof(int64_t) // <size_t>width:
        raise MemoryError()  # pragma: no cover - only a key matrix larger than SIZE_MAX bytes reaches this line
    keys = <int64_t*>PyMem_Malloc(<size_t>(m + 1) * <size_t>width * sizeof(int64_t))
    if keys == NULL:
        raise MemoryError()
    ref_tok = <PyObject**>PyMem_Malloc((m + n) * sizeof(PyObject*))
    if ref_tok == NULL:
        PyMem_Free(keys)  # pragma: no cover - only an allocator failure reaches this line
        raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
    hyp_tok = ref_tok + m

    try:
        _canonical_tokens(reference_word, hypothesis_word, ref_tok, hyp_tok)

        # Initialize first row and first column (boundary conditions)
        for j in range(n + 1):
            keys[j] = edit_step * j
        for i in range(1, m + 1):
            keys[i * width] = edit_step * i

        # Fill the key matrix
        for i in range(1, m + 1):
            left = keys[i * width]
            diag = keys[(i - 1) * width]
            for j in range(1, n + 1):
                step = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else sub_step

                up = keys[(i - 1) * width + j]
                best = up + edit_step
                if left + edit_step < best:
                    best = left + edit_step
                if diag + step < best:
                    best = diag + step

                keys[i * width + j] = best
                left = best
                diag = up

        insertions, deletions, substitutions = 0, 0, 0
        inserted_words, deleted_words, substituted_words = [], [], []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                same = ref_tok[i - 1] == hyp_tok[j - 1]
                if keys[i * width + j] == keys[(i - 1) * width + j - 1] + (0 if same else sub_step):
                    if not same:
                        substitutions += 1
                        substituted_words.append((reference_word[i - 1], hypothesis_word[j - 1]))
                    i -= 1
                    j -= 1
                    continue
            if j > 0 and keys[i * width + j] == keys[i * width + j - 1] + edit_step:
                insertions += 1
                inserted_words.append(hypothesis_word[j - 1])
                j -= 1
            elif i > 0 and keys[i * width + j] == keys[(i - 1) * width + j] + edit_step:
                deletions += 1
                deleted_words.append(reference_word[i - 1])
                i -= 1
            else:
                raise AssertionError("No move keeps the key optimal.")  # pragma: no cover - a move above keeps the key
    finally:
        PyMem_Free(ref_tok)
        PyMem_Free(keys)

    ld = insertions + deletions + substitutions
    wer = (<double>ld) / m if m > 0 else 0.0

    inserted_words.reverse(), deleted_words.reverse(), substituted_words.reverse()

    return np.array(
        [wer, ld, m, insertions, deletions, substitutions, inserted_words, deleted_words, substituted_words],
        dtype=object)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _metrics_batch_weighted(list references, list hypotheses, int order):
    """
    Batch processing for calculations_weighted.

    Returns (n, 9) object array where each row contains:
    [wer, ld, m, insertions, deletions, substitutions, inserted_words, deleted_words, substituted_words]
    """
    # The loop below reads both lists at every index without bounds checks.
    _check_equal_lengths(references, hypotheses)

    cdef Py_ssize_t n = len(references)
    cdef Py_ssize_t idx

    # Rows output, dtype=object because cols 6-8 are lists
    cdef object out = np.empty((n, 9), dtype=object)

    for idx in range(n):
        out[idx, :] = calculations_weighted(references[idx], hypotheses[idx], order)

    return out


cpdef object metrics_weighted(object reference, object hypothesis, int order):
    """
    Entry point of summaryp for weights that decide between the alignments with the fewest edits.

    Returns:
    - strings: a single row (len 9)
    - sequences: an (n, 9) object ndarray, one row per pair
    """
    _check_order(order)
    if isinstance(reference, (list, np.ndarray)) and isinstance(hypothesis, (list, np.ndarray)):
        return _metrics_batch_weighted(list(reference), list(hypothesis), order)
    return calculations_weighted(reference, hypothesis, order)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object calculations_fast_weighted(object reference, object hypothesis, int order):
    """
    Counts-only path for weights that decide between the alignments with the fewest edits.

    Among those alignments, it counts the one with the fewest substitutions when order is 1 and
    the one with the most substitutions when order is -1. The key of the last cell gives the
    edits and the substitutions, and the insertions and deletions follow from them and from the
    two lengths, so two rows of keys are kept and no backtrace is needed.

    Returns (6,) float64 array: [wer, ld, m, insertions, deletions, substitutions]
    """
    _check_order(order)

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

    cdef int64_t edit_step, sub_step, step, best, left, diag, up, key
    cdef int64_t ld, substitutions, rest
    cdef double wer

    # Two rows of keys in one buffer, swapped by pointer after each row
    cdef int64_t* rows
    cdef int64_t* prev
    cdef int64_t* curr
    cdef int64_t* tmp

    # Canonical token pointers, reference tokens first, hypothesis tokens after them
    cdef PyObject** ref_tok
    cdef PyObject** hyp_tok

    edit_step = (m if m < n else n) + 1
    sub_step = edit_step + order

    rows = <int64_t*>PyMem_Malloc(2 * (n + 1) * sizeof(int64_t))
    if rows == NULL:
        raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
    ref_tok = <PyObject**>PyMem_Malloc((m + n) * sizeof(PyObject*))
    if ref_tok == NULL:
        PyMem_Free(rows)  # pragma: no cover - only an allocator failure reaches this line
        raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
    hyp_tok = ref_tok + m
    prev = rows
    curr = rows + n + 1

    try:
        _canonical_tokens(reference_word, hypothesis_word, ref_tok, hyp_tok)

        for j in range(n + 1):
            prev[j] = edit_step * j

        for i in range(1, m + 1):
            curr[0] = edit_step * i
            left = curr[0]
            diag = prev[0]
            for j in range(1, n + 1):
                step = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else sub_step

                up = prev[j]
                best = up + edit_step
                if left + edit_step < best:
                    best = left + edit_step
                if diag + step < best:
                    best = diag + step

                curr[j] = best
                left = best
                diag = up

            tmp = prev
            prev = curr
            curr = tmp

        key = prev[n]
    finally:
        PyMem_Free(ref_tok)
        PyMem_Free(rows)

    # The edits and the substitutions of the alignment, from the key of the last cell
    if order > 0:
        ld = key // edit_step
        substitutions = key - edit_step * ld
    else:
        ld = (key + edit_step - 1) // edit_step
        substitutions = edit_step * ld - key
    rest = ld - substitutions
    wer = (<double>ld) / m if m > 0 else 0.0

    return np.array(
        [wer, <double>ld, <double>m,
         <double>((rest + n - m) // 2), <double>((rest - n + m) // 2), <double>substitutions],
        dtype=np.float64
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _metrics_batch_fast_weighted(list references, list hypotheses, int order):
    """
    Batch processing for calculations_fast_weighted.

    Returns (n, 6) float64 array where each row contains:
    [wer, ld, m, insertions, deletions, substitutions]
    """
    # The loop below reads both lists at every index without bounds checks.
    _check_equal_lengths(references, hypotheses)

    cdef Py_ssize_t n = len(references)
    cdef Py_ssize_t idx

    cdef object out = np.empty((n, 6), dtype=np.float64)

    for idx in range(n):
        out[idx, :] = calculations_fast_weighted(references[idx], hypotheses[idx], order)

    return out


cpdef object metrics_fast_weighted(object reference, object hypothesis, int order):
    """
    Entry point of werp and werps for weights that decide between the alignments with the fewest edits.

    Returns:
    - strings: (6,) float64 array [wer, ld, m, insertions, deletions, substitutions]
    - sequences: (n, 6) float64 array, one row per pair
    """
    _check_order(order)
    if isinstance(reference, (list, np.ndarray)) and isinstance(hypothesis, (list, np.ndarray)):
        return _metrics_batch_fast_weighted(list(reference), list(hypothesis), order)
    return calculations_fast_weighted(reference, hypothesis, order)
