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

calculations and calculations_fast take the same alignment, and their memory grows with the
number of words. For a short pair they keep the whole table of the dynamic programme and walk
back through it. For a longer pair they keep only the move that this backtrace takes at each
cell, in two bits. When the moves of a pair do not fit in a fixed size, the pair is split at
the cell where the backtrace leaves its middle reference row, and each part is aligned in the
same way. The counts and the word lists are the same in each case. table_budget() sets the two
sizes, so that the tests can reach the moves and the parts with short texts.

Input validation is performed by werpy._errorhandler.error_handler before these functions
are called by the public werpy functions.
"""

import numpy as np

cimport cython
from cpython.ref cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdint cimport int32_t, uint64_t

cdef extern from "_dictcompat.h":
    PyObject* werpy_dict_setdefault(object p, object key, object default) except NULL

# calculations() and calculations_fast() keep the whole table of the dynamic programme when it
# holds at most _table_bytes bytes. For a longer pair they keep only the move of the backtrace at
# each cell, in two bits, and align the pair in pieces whose moves fit in _code_bytes bytes
# (_align_pieces). The results are the same in every case.
cdef Py_ssize_t _table_bytes = 1048576
cdef Py_ssize_t _code_bytes = 67108864

# The moves of the backtrace, in the order in which it prefers them
cdef enum:
    _DIAGONAL = 0
    _INSERTION = 1
    _DELETION = 2


def table_budget(Py_ssize_t table=-1, Py_ssize_t codes=-1):
    """
    Return the largest whole table and the largest table of moves, in bytes, that calculations()
    and calculations_fast() keep, and set them to table and codes when these are 0 or more. The
    results do not depend on these sizes; only the memory and the time do. For the tests.
    """
    global _table_bytes, _code_bytes
    previous = (_table_bytes, _code_bytes)
    if table >= 0:
        _table_bytes = table
    if codes >= 0:
        _code_bytes = codes
    return previous


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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _align_table(
    PyObject** ref_tok,
    PyObject** hyp_tok,
    Py_ssize_t m,
    Py_ssize_t n,
    int32_t* ldm,
    list reference_word,
    list hypothesis_word,
    list inserted_words,
    list deleted_words,
    list substituted_words,
    int* counts,
) except -1:
    """
    Align the m reference words with the n hypothesis words through the whole (m + 1) x (n + 1)
    Levenshtein distance matrix in ldm, row by row. The backtrace walks from (m, n) back to
    (0, 0) and adds its insertions, deletions and substitutions to counts[0], counts[1] and
    counts[2]. When the word lists are not None, it also appends the words to them in the order
    of the walk.
    """
    cdef Py_ssize_t width = n + 1
    cdef Py_ssize_t i, j
    cdef int cost, del_cost, ins_cost, sub_cost, best
    cdef int left, diag, up
    cdef int32_t* row
    cdef int32_t* above

    # Initialize first row and first column (boundary conditions)
    for j in range(n + 1):
        ldm[j] = <int32_t>j
    for i in range(1, m + 1):
        ldm[i * width] = <int32_t>i

    # Fill the Levenshtein distance matrix
    # Compute edit distances using a branch-free inner loop and manual minimum
    # selection to keep all operations at C level and minimize per-cell overhead.
    # No boundary condition branches in the hot loop, manual min selection.
    for i in range(1, m + 1):
        row = ldm + i * width
        above = row - width
        left = row[0]
        diag = above[0]
        for j in range(1, n + 1):
            cost = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else 1

            up = above[j]
            del_cost = up + 1
            ins_cost = left + 1
            sub_cost = diag + cost

            best = del_cost
            if ins_cost < best:
                best = ins_cost
            if sub_cost < best:
                best = sub_cost

            row[j] = best
            left = best
            diag = up

    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tok[i - 1] == hyp_tok[j - 1]:
            i -= 1
            j -= 1
        else:
            if i > 0 and j > 0 and ldm[i * width + j] == ldm[(i - 1) * width + j - 1] + 1:
                counts[2] += 1
                if substituted_words is not None:
                    substituted_words.append((reference_word[i - 1], hypothesis_word[j - 1]))
                i -= 1
                j -= 1
            elif j > 0 and ldm[i * width + j] == ldm[i * width + j - 1] + 1:
                counts[0] += 1
                if inserted_words is not None:
                    inserted_words.append(hypothesis_word[j - 1])
                j -= 1
            elif i > 0 and ldm[i * width + j] == ldm[(i - 1) * width + j] + 1:
                counts[1] += 1
                if deleted_words is not None:
                    deleted_words.append(reference_word[i - 1])
                i -= 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _align_codes(
    PyObject** ref_tok,
    PyObject** hyp_tok,
    Py_ssize_t m,
    Py_ssize_t n,
    int32_t* rows,
    uint64_t* codes,
    list reference_word,
    list hypothesis_word,
    Py_ssize_t ref_start,
    Py_ssize_t hyp_start,
    list inserted_words,
    list deleted_words,
    list substituted_words,
    int* counts,
) except -1:
    """
    Align as _align_table does, keeping two rows of the matrix in rows (2 (n + 1) cells) and, for
    each cell of rows 1 to m and columns 1 to n, only the move that the backtrace takes there, in
    two bits of codes (m rows of (n + 31) // 32 64-bit words). The move is the first of the
    diagonal, an insertion and a deletion whose cell plus the cost of the move gives the value of
    the cell, which is the test of the backtrace of _align_table. The walk then reads the moves.
    The words start at ref_start and hyp_start in the word lists.
    """
    cdef Py_ssize_t words = (n + 31) // 32
    cdef Py_ssize_t i, j, shift
    cdef int cost, best, move
    cdef int left, diag, up
    cdef int32_t* prev = rows
    cdef int32_t* curr = rows + n + 1
    cdef int32_t* tmp
    cdef uint64_t* out
    cdef uint64_t acc

    for j in range(n + 1):
        prev[j] = <int32_t>j
    for i in range(1, m + 1):
        curr[0] = <int32_t>i
        left = <int>i
        diag = prev[0]
        out = codes + (i - 1) * words
        acc = 0
        shift = 0
        for j in range(1, n + 1):
            cost = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else 1

            up = prev[j]
            best = up + 1
            move = _DELETION
            if left + 1 <= best:
                best = left + 1
                move = _INSERTION
            if diag + cost <= best:
                best = diag + cost
                move = _DIAGONAL

            curr[j] = best
            acc |= (<uint64_t>move) << shift
            shift += 2
            if shift == 64:
                out[0] = acc
                out += 1
                acc = 0
                shift = 0
            left = best
            diag = up
        if shift:
            out[0] = acc
        tmp = prev
        prev = curr
        curr = tmp

    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            move = _INSERTION
        elif j == 0:
            move = _DELETION
        else:
            move = <int>((codes[(i - 1) * words + ((j - 1) >> 5)] >> (2 * ((j - 1) & 31))) & 3)
        if move == _DIAGONAL:
            if ref_tok[i - 1] != hyp_tok[j - 1]:
                counts[2] += 1
                if substituted_words is not None:
                    substituted_words.append((reference_word[ref_start + i - 1], hypothesis_word[hyp_start + j - 1]))
            i -= 1
            j -= 1
        elif move == _INSERTION:
            counts[0] += 1
            if inserted_words is not None:
                inserted_words.append(hypothesis_word[hyp_start + j - 1])
            j -= 1
        else:
            counts[1] += 1
            if deleted_words is not None:
                deleted_words.append(reference_word[ref_start + i - 1])
            i -= 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t _split_column(
    PyObject** ref_tok,
    PyObject** hyp_tok,
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t r,
    int32_t* prev,
    int32_t* curr,
    Py_ssize_t* cols_prev,
    Py_ssize_t* cols_curr,
) noexcept:
    """
    Return the column at which the backtrace of _align_table, walking from (m, n), leaves row r,
    for 1 <= r <= m: the column of its last cell in that row. The rows of the matrix are computed
    one by one in prev and curr, n + 1 cells each. From row r on, each cell also takes the move
    that the backtrace takes there: the first of the diagonal, an insertion and a deletion whose
    cell plus the cost of the move gives the value of the cell. With that move, each cell gets the
    column at which the backtrace from it leaves row r, in cols_prev and cols_curr.
    """
    cdef Py_ssize_t i, j
    cdef int cost, del_cost, ins_cost, sub_cost, best, move
    cdef int left, diag, up
    cdef int32_t* tmp
    cdef Py_ssize_t* cols_tmp

    for j in range(n + 1):
        prev[j] = <int32_t>j
    for i in range(1, m + 1):
        curr[0] = <int32_t>i
        left = <int>i
        diag = prev[0]
        if i < r:
            for j in range(1, n + 1):
                cost = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else 1

                up = prev[j]
                del_cost = up + 1
                ins_cost = left + 1
                sub_cost = diag + cost

                best = del_cost
                if ins_cost < best:
                    best = ins_cost
                if sub_cost < best:
                    best = sub_cost

                curr[j] = best
                left = best
                diag = up
        else:
            # Column 0 is reached by deletions only, so the backtrace from it leaves row r at column 0
            cols_curr[0] = 0
            for j in range(1, n + 1):
                cost = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else 1

                up = prev[j]
                best = up + 1
                move = _DELETION
                if left + 1 <= best:
                    best = left + 1
                    move = _INSERTION
                if diag + cost <= best:
                    best = diag + cost
                    move = _DIAGONAL

                curr[j] = best
                if move == _INSERTION:
                    cols_curr[j] = cols_curr[j - 1]
                elif i == r:
                    cols_curr[j] = j
                elif move == _DIAGONAL:
                    cols_curr[j] = cols_prev[j - 1]
                else:
                    cols_curr[j] = cols_prev[j]
                left = best
                diag = up
            cols_tmp = cols_prev
            cols_prev = cols_curr
            cols_curr = cols_tmp
        tmp = prev
        prev = curr
        curr = tmp
    return cols_prev[n]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _align_pieces(
    PyObject** ref_tok,
    PyObject** hyp_tok,
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t code_bytes,
    list reference_word,
    list hypothesis_word,
    list inserted_words,
    list deleted_words,
    list substituted_words,
    int* counts,
) except -1:
    """
    Align the texts as _align_table does, keeping only the moves of the backtrace. A piece whose
    moves fit in code_bytes bytes, or that has at most one reference word, is aligned by
    _align_codes; the whole pair is the first piece. Any other piece is split at its middle
    reference row r (Hirschberg's divide and conquer): the backtrace leaves row r at the column c
    that _split_column returns, so it is the backtrace of the piece from (r, c) to the end, then
    that of the piece from the start to (r, c), each over a matrix of its own words. The pieces
    are aligned from the end of the texts, so the words are appended in the order of the
    backtrace. Memory: the moves of the largest piece, which is at most code_bytes bytes or one
    reference word, and four rows of n + 1 cells.
    """
    # A split halves the reference words of a piece, so the stack holds fewer than 68 pieces (a, b, c, d)
    cdef Py_ssize_t stack[272]
    cdef Py_ssize_t top
    cdef Py_ssize_t a, b, c, d, h, w, r, col, words, limit, half, capacity
    cdef int32_t* rows
    cdef Py_ssize_t* cols
    cdef uint64_t* codes

    # The moves of the largest piece that _align_codes aligns: the whole pair when its moves fit.
    # Otherwise a split leaves pieces of at most half the reference words, rounded up, and n
    # hypothesis words, and each of them is aligned when its moves fit or it has one reference word.
    words = (n + 31) // 32
    limit = code_bytes // <Py_ssize_t>sizeof(uint64_t)
    if m <= 1 or words == 0 or m <= limit // words:
        capacity = m * words
    else:
        half = (m + 1) // 2
        capacity = half * words if half <= limit // words else limit
        if capacity < words:
            capacity = words

    rows = <int32_t*>PyMem_Malloc(2 * (n + 1) * sizeof(int32_t))
    if rows == NULL:
        raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
    cols = <Py_ssize_t*>PyMem_Malloc(2 * (n + 1) * sizeof(Py_ssize_t))
    if cols == NULL:
        PyMem_Free(rows)  # pragma: no cover - only an allocator failure reaches this line
        raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
    codes = <uint64_t*>PyMem_Malloc(capacity * sizeof(uint64_t))
    if codes == NULL:
        PyMem_Free(cols)  # pragma: no cover - only an allocator failure reaches this line
        PyMem_Free(rows)  # pragma: no cover - only an allocator failure reaches this line
        raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line

    try:
        stack[0] = 0
        stack[1] = m
        stack[2] = 0
        stack[3] = n
        top = 4
        while top > 0:
            top -= 4
            a = stack[top]
            b = stack[top + 1]
            c = stack[top + 2]
            d = stack[top + 3]
            h = b - a
            w = d - c
            words = (w + 31) // 32
            if h <= 1 or words == 0 or h <= limit // words:
                _align_codes(ref_tok + a, hyp_tok + c, h, w, rows, codes, reference_word, hypothesis_word, a, c,
                             inserted_words, deleted_words, substituted_words, counts)
            else:
                r = h // 2
                col = _split_column(ref_tok + a, hyp_tok + c, h, w, r, rows, rows + n + 1, cols, cols + n + 1)
                # The piece from the start to (r, col) is aligned after the piece from (r, col) to the end
                stack[top] = a
                stack[top + 1] = a + r
                stack[top + 2] = c
                stack[top + 3] = c + col
                stack[top + 4] = a + r
                stack[top + 5] = b
                stack[top + 6] = c + col
                stack[top + 7] = d
                top += 8
    finally:
        PyMem_Free(codes)
        PyMem_Free(cols)
        PyMem_Free(rows)
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _align(
    list reference_word,
    list hypothesis_word,
    list inserted_words,
    list deleted_words,
    list substituted_words,
    int* counts,
) except -1:
    """
    Align the reference words with the hypothesis words, as calculations() and
    calculations_fast() do, and put the insertions, deletions and substitutions of the backtrace
    in counts[0], counts[1] and counts[2]. When the word lists are not None, the words are also
    appended to them in the order of the backtrace, from the end of the texts. The whole matrix is
    kept when it fits in _table_bytes bytes, and only the moves otherwise (_align_pieces).
    """
    cdef Py_ssize_t m = len(reference_word)
    cdef Py_ssize_t n = len(hypothesis_word)

    # The Levenshtein distance matrix, when the whole of it is kept
    cdef int32_t* ldm

    # Canonical token pointers, reference tokens first, hypothesis tokens after them
    cdef PyObject** ref_tok
    cdef PyObject** hyp_tok

    counts[0] = 0
    counts[1] = 0
    counts[2] = 0

    ref_tok = <PyObject**>PyMem_Malloc((m + n) * sizeof(PyObject*))
    if ref_tok == NULL:
        raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
    hyp_tok = ref_tok + m

    try:
        _canonical_tokens(reference_word, hypothesis_word, ref_tok, hyp_tok)

        # The whole (m+1) x (n+1) matrix when it fits in _table_bytes; only the moves otherwise
        if m + 1 <= _table_bytes // <Py_ssize_t>sizeof(int32_t) // (n + 1):
            ldm = <int32_t*>PyMem_Malloc((m + 1) * (n + 1) * sizeof(int32_t))
            if ldm == NULL:
                raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
            try:
                _align_table(ref_tok, hyp_tok, m, n, ldm, reference_word, hypothesis_word,
                             inserted_words, deleted_words, substituted_words, counts)
            finally:
                PyMem_Free(ldm)
        else:
            _align_pieces(ref_tok, hyp_tok, m, n, _code_bytes, reference_word, hypothesis_word,
                          inserted_words, deleted_words, substituted_words, counts)
    finally:
        PyMem_Free(ref_tok)
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object calculations(object reference, object hypothesis):
    cdef list reference_word = reference.split()
    cdef list hypothesis_word

    # Use Py_ssize_t for indices and sizes
    # Py_ssize_t matches Python's internal index type and avoids unnecessary
    # casts or overflow risks when working with Python lists.
    cdef Py_ssize_t m = len(reference_word)

    # An identical pair has no edits, so the token mapping and the dynamic
    # programme are skipped. Only the reference word count is needed.
    if reference == hypothesis:
        return np.array([0.0, 0, m, 0, 0, 0, [], [], []], dtype=object)

    hypothesis_word = hypothesis.split()

    # Metrics and outputs
    cdef int ld, insertions, deletions, substitutions
    cdef double wer
    cdef list inserted_words, deleted_words, substituted_words
    cdef int counts[3]

    inserted_words, deleted_words, substituted_words = [], [], []
    _align(reference_word, hypothesis_word, inserted_words, deleted_words, substituted_words, counts)

    insertions, deletions, substitutions = counts[0], counts[1], counts[2]
    ld = insertions + deletions + substitutions
    wer = (<double>ld) / m if m > 0 else 0.0

    inserted_words.reverse(), deleted_words.reverse(), substituted_words.reverse()

    return np.array(
        [wer, ld, m, insertions, deletions, substitutions, inserted_words, deleted_words, substituted_words],
        dtype=object)


cdef inline int _check_equal_lengths(list references, list hypotheses) except -1:
    """
    Raise a ValueError when references and hypotheses have different numbers of elements.
    """
    if len(references) != len(hypotheses):
        raise ValueError("The Reference and Hypothesis input parameters must have the same number of elements.")
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _metrics_batch(list references, list hypotheses):
    """
    Private batch processing function. Processes multiple reference-hypothesis
    pairs at C speed, eliminating np.vectorize overhead.

    Returns (n, 9) object array where each row contains:
    [wer, ld, m, insertions, deletions, substitutions, inserted_words, deleted_words, substituted_words]
    """
    # The loop below reads both lists at every index without bounds checks.
    _check_equal_lengths(references, hypotheses)

    cdef Py_ssize_t n = len(references)
    cdef Py_ssize_t idx

    # Rows output, dtype=object because cols 6-8 are lists
    cdef object out = np.empty((n, 9), dtype=object)

    cdef object r
    for idx in range(n):
        r = calculations(references[idx], hypotheses[idx])
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
cpdef object calculations_fast(object reference, object hypothesis):
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

    # An identical pair has no edits, so the token mapping and the dynamic
    # programme are skipped. Only the reference word count is needed.
    if reference == hypothesis:
        return np.array([0.0, 0.0, <double>m, 0.0, 0.0, 0.0], dtype=np.float64)

    hypothesis_word = hypothesis.split()

    cdef int ld, insertions, deletions, substitutions
    cdef double wer
    cdef int counts[3]

    # The alignment of calculations(); the backtrace counts the errors without word tracking
    _align(reference_word, hypothesis_word, None, None, None, counts)

    insertions, deletions, substitutions = counts[0], counts[1], counts[2]
    ld = insertions + deletions + substitutions
    wer = (<double>ld) / m if m > 0 else 0.0

    return np.array(
        [wer, <double>ld, <double>m,
         <double>insertions, <double>deletions, <double>substitutions],
        dtype=np.float64
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _metrics_batch_fast(list references, list hypotheses):
    """
    Fast batch processing without word tracking.

    Returns (n, 6) float64 array where each row contains:
    [wer, ld, m, insertions, deletions, substitutions]
    """
    # The loop below reads both lists at every index without bounds checks.
    _check_equal_lengths(references, hypotheses)

    cdef Py_ssize_t n = len(references)
    cdef Py_ssize_t idx

    cdef object out = np.empty((n, 6), dtype=np.float64)

    cdef object r
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
cpdef object calculations_wer_only(object reference, object hypothesis):
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
    cdef int left, diag, up
    cdef double wer

    # Canonical token pointers, reference tokens first, hypothesis tokens after them
    cdef PyObject** ref_tok
    cdef PyObject** hyp_tok

    cdef object prev_arr
    cdef object curr_arr
    cdef int32_t[:] prev
    cdef int32_t[:] curr

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
        prev[j] = <int32_t>j

    ref_tok = <PyObject**>PyMem_Malloc((m + n) * sizeof(PyObject*))
    if ref_tok == NULL:
        raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
    hyp_tok = ref_tok + m

    try:
        _canonical_tokens(reference_word, hypothesis_word, ref_tok, hyp_tok)

        for i in range(1, m + 1):
            curr[0] = <int32_t>i
            left = <int>i
            diag = prev[0]
            for j in range(1, n + 1):
                cost = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else 1

                up = prev[j]
                del_cost = up + 1
                ins_cost = left + 1
                sub_cost = diag + cost

                best = del_cost
                if ins_cost < best:
                    best = ins_cost
                if sub_cost < best:
                    best = sub_cost

                curr[j] = best
                left = best
                diag = up

            prev, curr = curr, prev
    finally:
        PyMem_Free(ref_tok)

    ld = prev[n]
    wer = (<double>ld) / m if m > 0 else 0.0

    return np.array([wer, <double>ld, <double>m], dtype=np.float64)


cdef struct _BatchBuffers:
    # Two dynamic programming rows and one token buffer shared across a batch.
    # Each is grown on demand and freed once when the batch is finished.
    int32_t* prev
    int32_t* curr
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
    cdef int32_t* row_mem
    cdef PyObject** tok_mem

    if rows > buf.row_capacity:
        capacity = 2 * buf.row_capacity
        if capacity < rows:
            capacity = rows
        row_mem = <int32_t*>PyMem_Realloc(buf.prev, 2 * capacity * sizeof(int32_t))
        if row_mem == NULL:
            raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
        buf.prev = row_mem
        buf.curr = row_mem + capacity
        buf.row_capacity = capacity

    if tokens > buf.tok_capacity:
        capacity = 2 * buf.tok_capacity
        if capacity < tokens:
            capacity = tokens
        tok_mem = <PyObject**>PyMem_Realloc(buf.tok, capacity * sizeof(PyObject*))
        if tok_mem == NULL:
            raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
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
    cdef int left, diag, up
    cdef int32_t* prev
    cdef int32_t* curr
    cdef int32_t* tmp

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
        prev[j] = <int32_t>j

    _canonical_tokens(reference_word, hypothesis_word, ref_tok, hyp_tok)

    for i in range(1, m + 1):
        curr[0] = <int32_t>i
        left = <int>i
        diag = prev[0]
        for j in range(1, n + 1):
            cost = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else 1

            up = prev[j]
            del_cost = up + 1
            ins_cost = left + 1
            sub_cost = diag + cost

            best = del_cost
            if ins_cost < best:
                best = ins_cost
            if sub_cost < best:
                best = sub_cost

            curr[j] = best
            left = best
            diag = up

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
cdef object _metrics_batch_wer_only(list references, list hypotheses):
    """
    Fast batch processing for WER-only calculations with buffer reuse and pointer swapping.

    One pair of dynamic programming rows and one token buffer are shared by every pair in
    the batch. They are grown on demand as longer pairs are met and freed once at the end,
    so the hypotheses are split only inside the loop. Uses true pointer swapping instead
    of value copying for optimal performance.

    Returns (n, 3) float64 array where each row contains:
    [wer, ld, m]
    """
    # The loop below reads both lists at every index without bounds checks.
    _check_equal_lengths(references, hypotheses)

    cdef Py_ssize_t n_pairs = len(references)
    cdef Py_ssize_t idx

    cdef object out = np.empty((n_pairs, 3), dtype=np.float64)
    cdef double[:, ::1] out_view
    cdef double* out_data

    # An empty batch returns the empty output array without taking a view of it.
    if n_pairs == 0:
        return out

    out_view = out
    out_data = &out_view[0, 0]

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
