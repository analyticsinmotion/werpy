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

- calculations_weighted(reference, hypothesis, order) -> tuple: the full path. Returns the
  tuple of nine items of werpy._metrics.calculations for that alignment: the WER, LD, number
  of words in the reference, counts of insertions, deletions and substitutions, and lists of
  the inserted, deleted and substituted words.
- metrics_weighted(reference, hypothesis, order) -> tuple or list: routes to
  calculations_weighted for a pair of strings, or processes a batch and returns a list with
  one such tuple for each pair.
- calculations_fast_weighted(reference, hypothesis, order) -> tuple: the counts-only path.
  Returns the tuple of six floats of werpy._metrics.calculations_fast for that alignment: the
  WER, LD, number of words in the reference, and counts of insertions, deletions and
  substitutions. It keeps two rows of keys and has no backtrace.
- metrics_fast_weighted(reference, hypothesis, order) -> tuple or list: routes to
  calculations_fast_weighted for a pair of strings, or returns a list with one such tuple for
  each pair of a batch.

The memory of calculations_weighted grows with the number of words. For a short pair it keeps
the whole key matrix and walks back through it. For a longer pair it keeps only the move that
this backtrace takes at each cell, in two bits. When the moves of a pair do not fit in a fixed
size, the pair is split at the cell where the backtrace leaves its middle reference row, and
each part is aligned in the same way, with the steps of the whole pair. The counts and the word
lists are the same in each case. table_budget() sets the two sizes, so that the tests can reach
the moves and the parts with short texts.

Each function raises a ValueError when the order is not 1 or -1.

Input validation is performed by werpy._errorhandler.error_handler before these functions
are called by the public werpy functions.
"""

cimport cython
from cpython.ref cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport int64_t, uint64_t

cdef extern from "_dictcompat.h":
    PyObject* werpy_dict_setdefault(object p, object key, object default) except NULL

# calculations_weighted() keeps the whole key matrix when it holds at most _table_bytes bytes. For
# a longer pair it keeps only the move of the backtrace at each cell, in two bits, and aligns the
# pair in pieces whose moves fit in _code_bytes bytes (_align_pieces). The result is the same in
# every case.
cdef Py_ssize_t _table_bytes = 1048576
cdef Py_ssize_t _code_bytes = 67108864

# The moves of the backtrace, in the order in which it prefers them
cdef enum:
    _DIAGONAL = 0
    _INSERTION = 1
    _DELETION = 2


def table_budget(Py_ssize_t table=-1, Py_ssize_t codes=-1):
    """
    Return the largest whole key matrix and the largest table of moves, in bytes, that
    calculations_weighted() keeps, and set them to table and codes when these are 0 or more. The
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
cdef int _align_table(
    PyObject** ref_tok,
    PyObject** hyp_tok,
    Py_ssize_t m,
    Py_ssize_t n,
    int64_t edit_step,
    int64_t sub_step,
    int64_t* keys,
    list reference_word,
    list hypothesis_word,
    list inserted_words,
    list deleted_words,
    list substituted_words,
    int* counts,
) except -1:
    """
    Align the m reference words with the n hypothesis words through the whole (m + 1) x (n + 1)
    key matrix in keys, row by row. The backtrace walks from (m, n) back to (0, 0), adds its
    insertions, deletions and substitutions to counts[0], counts[1] and counts[2], and appends the
    words to the lists in the order of the walk.
    """
    cdef Py_ssize_t width = n + 1
    cdef Py_ssize_t i, j
    cdef int64_t step, best, left, diag, up
    cdef int64_t* row
    cdef int64_t* above
    cdef bint same

    # Initialize first row and first column (boundary conditions)
    for j in range(n + 1):
        keys[j] = edit_step * j
    for i in range(1, m + 1):
        keys[i * width] = edit_step * i

    # Fill the key matrix
    for i in range(1, m + 1):
        row = keys + i * width
        above = row - width
        left = row[0]
        diag = above[0]
        for j in range(1, n + 1):
            step = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else sub_step

            up = above[j]
            best = up + edit_step
            if left + edit_step < best:
                best = left + edit_step
            if diag + step < best:
                best = diag + step

            row[j] = best
            left = best
            diag = up

    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = ref_tok[i - 1] == hyp_tok[j - 1]
            if keys[i * width + j] == keys[(i - 1) * width + j - 1] + (0 if same else sub_step):
                if not same:
                    counts[2] += 1
                    substituted_words.append((reference_word[i - 1], hypothesis_word[j - 1]))
                i -= 1
                j -= 1
                continue
        if j > 0 and keys[i * width + j] == keys[i * width + j - 1] + edit_step:
            counts[0] += 1
            inserted_words.append(hypothesis_word[j - 1])
            j -= 1
        elif i > 0 and keys[i * width + j] == keys[(i - 1) * width + j] + edit_step:
            counts[1] += 1
            deleted_words.append(reference_word[i - 1])
            i -= 1
        else:
            raise AssertionError("No move keeps the key optimal.")  # pragma: no cover - a move above keeps the key
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _align_codes(
    PyObject** ref_tok,
    PyObject** hyp_tok,
    Py_ssize_t m,
    Py_ssize_t n,
    int64_t edit_step,
    int64_t sub_step,
    int64_t* rows,
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
    Align as _align_table does, keeping two rows of the key matrix in rows (2 (n + 1) cells) and,
    for each cell of rows 1 to m and columns 1 to n, only the move that the backtrace takes there,
    in two bits of codes (m rows of (n + 31) // 32 64-bit words). The move is the first of the
    diagonal, an insertion and a deletion whose cell plus the step of the move gives the key of
    the cell, which is the test of the backtrace of _align_table. The walk then reads the moves.
    The words start at ref_start and hyp_start in the word lists.
    """
    cdef Py_ssize_t words = (n + 31) // 32
    cdef Py_ssize_t i, j, shift
    cdef int64_t step, best, left, diag, up
    cdef int move
    cdef int64_t* prev = rows
    cdef int64_t* curr = rows + n + 1
    cdef int64_t* tmp
    cdef uint64_t* out
    cdef uint64_t acc

    for j in range(n + 1):
        prev[j] = edit_step * j
    for i in range(1, m + 1):
        curr[0] = edit_step * i
        left = curr[0]
        diag = prev[0]
        out = codes + (i - 1) * words
        acc = 0
        shift = 0
        for j in range(1, n + 1):
            step = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else sub_step

            up = prev[j]
            best = up + edit_step
            move = _DELETION
            if left + edit_step <= best:
                best = left + edit_step
                move = _INSERTION
            if diag + step <= best:
                best = diag + step
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
                substituted_words.append((reference_word[ref_start + i - 1], hypothesis_word[hyp_start + j - 1]))
            i -= 1
            j -= 1
        elif move == _INSERTION:
            counts[0] += 1
            inserted_words.append(hypothesis_word[hyp_start + j - 1])
            j -= 1
        else:
            counts[1] += 1
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
    int64_t edit_step,
    int64_t sub_step,
    Py_ssize_t r,
    int64_t* prev,
    int64_t* curr,
    Py_ssize_t* cols_prev,
    Py_ssize_t* cols_curr,
) noexcept:
    """
    Return the column at which the backtrace of _align_table, walking from (m, n), leaves row r,
    for 1 <= r <= m: the column of its last cell in that row. The rows of the key matrix are
    computed one by one in prev and curr, n + 1 cells each. From row r on, each cell also takes
    the move that the backtrace takes there: the first of the diagonal, an insertion and a
    deletion whose cell plus the step of the move gives the key of the cell. With that move, each
    cell gets the column at which the backtrace from it leaves row r, in cols_prev and cols_curr.
    """
    cdef Py_ssize_t i, j
    cdef int64_t step, best, left, diag, up
    cdef int move
    cdef int64_t* tmp
    cdef Py_ssize_t* cols_tmp

    for j in range(n + 1):
        prev[j] = edit_step * j
    for i in range(1, m + 1):
        curr[0] = edit_step * i
        left = curr[0]
        diag = prev[0]
        if i < r:
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
        else:
            # Column 0 is reached by deletions only, so the backtrace from it leaves row r at column 0
            cols_curr[0] = 0
            for j in range(1, n + 1):
                step = 0 if ref_tok[i - 1] == hyp_tok[j - 1] else sub_step

                up = prev[j]
                best = up + edit_step
                move = _DELETION
                if left + edit_step <= best:
                    best = left + edit_step
                    move = _INSERTION
                if diag + step <= best:
                    best = diag + step
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
    int64_t edit_step,
    int64_t sub_step,
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
    that of the piece from the start to (r, c), each over a key matrix of its own words with the
    steps of the whole pair. The pieces are aligned from the end of the texts, so the words are
    appended in the order of the backtrace. Memory: the moves of the largest piece, which is at
    most code_bytes bytes or one reference word, and four rows of n + 1 cells.
    """
    # A split halves the reference words of a piece, so the stack holds fewer than 68 pieces (a, b, c, d)
    cdef Py_ssize_t stack[272]
    cdef Py_ssize_t top
    cdef Py_ssize_t a, b, c, d, h, w, r, col, words, limit, half, capacity
    cdef int64_t* rows
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

    rows = <int64_t*>PyMem_Malloc(2 * (n + 1) * sizeof(int64_t))
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
                _align_codes(ref_tok + a, hyp_tok + c, h, w, edit_step, sub_step, rows, codes, reference_word,
                             hypothesis_word, a, c, inserted_words, deleted_words, substituted_words, counts)
            else:
                r = h // 2
                col = _split_column(ref_tok + a, hyp_tok + c, h, w, edit_step, sub_step, r, rows, rows + n + 1,
                                    cols, cols + n + 1)
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
cpdef object calculations_weighted(object reference, object hypothesis, int order):
    """
    Full path for weights that decide between the alignments with the fewest edits.

    Among those alignments, it takes the one with the fewest substitutions when order is 1 and
    the one with the most substitutions when order is -1. The backtrace reads the texts from the
    end and takes, at each cell, the first move that keeps the key optimal: a match or a
    substitution, then an insertion, then a deletion, as in werpy._metrics.calculations.

    Returns a tuple of 9 items: (wer, ld, m, insertions, deletions, substitutions,
    inserted_words, deleted_words, substituted_words)
    """
    _check_order(order)

    cdef list reference_word = reference.split()
    cdef list hypothesis_word

    cdef Py_ssize_t m = len(reference_word)
    cdef Py_ssize_t n

    # An identical pair has no edits, so the token mapping and the dynamic
    # programme are skipped. Only the reference word count is needed.
    if reference == hypothesis:
        return (0.0, 0, m, 0, 0, 0, [], [], [])

    hypothesis_word = hypothesis.split()
    n = len(hypothesis_word)

    # Metrics and outputs
    cdef int ld, insertions, deletions, substitutions
    cdef double wer
    cdef list inserted_words, deleted_words, substituted_words
    cdef int counts[3]

    # Variables of the dynamic programme. edit_step is larger than any number of
    # substitutions, so a key orders alignments by their edits first.
    cdef int64_t edit_step, sub_step
    cdef int64_t* keys

    # Canonical token pointers, reference tokens first, hypothesis tokens after them
    cdef PyObject** ref_tok
    cdef PyObject** hyp_tok

    edit_step = (m if m < n else n) + 1
    sub_step = edit_step + order

    ref_tok = <PyObject**>PyMem_Malloc((m + n) * sizeof(PyObject*))
    if ref_tok == NULL:
        raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
    hyp_tok = ref_tok + m

    counts[0] = 0
    counts[1] = 0
    counts[2] = 0
    inserted_words, deleted_words, substituted_words = [], [], []
    try:
        _canonical_tokens(reference_word, hypothesis_word, ref_tok, hyp_tok)

        # The whole (m+1) x (n+1) key matrix when it fits in _table_bytes; only the moves otherwise
        if m + 1 <= _table_bytes // <Py_ssize_t>sizeof(int64_t) // (n + 1):
            keys = <int64_t*>PyMem_Malloc((m + 1) * (n + 1) * sizeof(int64_t))
            if keys == NULL:
                raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
            try:
                _align_table(ref_tok, hyp_tok, m, n, edit_step, sub_step, keys, reference_word, hypothesis_word,
                             inserted_words, deleted_words, substituted_words, counts)
            finally:
                PyMem_Free(keys)
        else:
            _align_pieces(ref_tok, hyp_tok, m, n, edit_step, sub_step, _code_bytes, reference_word,
                          hypothesis_word, inserted_words, deleted_words, substituted_words, counts)
    finally:
        PyMem_Free(ref_tok)

    insertions, deletions, substitutions = counts[0], counts[1], counts[2]
    ld = insertions + deletions + substitutions
    wer = (<double>ld) / m if m > 0 else 0.0

    inserted_words.reverse(), deleted_words.reverse(), substituted_words.reverse()

    return (wer, ld, m, insertions, deletions, substitutions, inserted_words, deleted_words, substituted_words)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _metrics_batch_weighted(list references, list hypotheses, int order):
    """
    Batch processing for calculations_weighted.

    Returns a list with one tuple for each pair:
    (wer, ld, m, insertions, deletions, substitutions, inserted_words, deleted_words, substituted_words)
    """
    # The loop below reads both lists at every index without bounds checks.
    _check_equal_lengths(references, hypotheses)

    cdef Py_ssize_t n = len(references)
    cdef Py_ssize_t idx

    # One row for each pair
    cdef list out = []

    for idx in range(n):
        out.append(calculations_weighted(references[idx], hypotheses[idx], order))

    return out


cpdef object metrics_weighted(object reference, object hypothesis, int order):
    """
    Entry point of summaryp for weights that decide between the alignments with the fewest edits.

    Returns:
    - strings: a single row, a tuple of 9 items
    - lists: a list with one row for each pair
    """
    _check_order(order)
    if isinstance(reference, list) and isinstance(hypothesis, list):
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

    Returns a tuple of 6 floats: (wer, ld, m, insertions, deletions, substitutions)
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
        return (0.0, 0.0, <double>m, 0.0, 0.0, 0.0)

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

    return (wer, <double>ld, <double>m,
            <double>((rest + n - m) // 2), <double>((rest - n + m) // 2), <double>substitutions)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _metrics_batch_fast_weighted(list references, list hypotheses, int order):
    """
    Batch processing for calculations_fast_weighted.

    Returns a list with one tuple of 6 floats for each pair:
    (wer, ld, m, insertions, deletions, substitutions)
    """
    # The loop below reads both lists at every index without bounds checks.
    _check_equal_lengths(references, hypotheses)

    cdef Py_ssize_t n = len(references)
    cdef Py_ssize_t idx

    # One row for each pair
    cdef list out = []

    for idx in range(n):
        out.append(calculations_fast_weighted(references[idx], hypotheses[idx], order))

    return out


cpdef object metrics_fast_weighted(object reference, object hypothesis, int order):
    """
    Entry point of werp and werps for weights that decide between the alignments with the fewest edits.

    Returns:
    - strings: a tuple of 6 floats (wer, ld, m, insertions, deletions, substitutions)
    - lists: a list with one such tuple for each pair
    """
    _check_order(order)
    if isinstance(reference, list) and isinstance(hypothesis, list):
        return _metrics_batch_fast_weighted(list(reference), list(hypothesis), order)
    return calculations_fast_weighted(reference, hypothesis, order)
