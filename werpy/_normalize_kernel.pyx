# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
This Cython module applies the rule of werpy.normalize to the characters of a text. werpy._normalize builds the
table that it reads from the Unicode database of the running Python, one block of 256 code points at a time, when a
text first holds a character of that block.

For each character the table gives whether the rule deletes it, its lowercase, whether the character and its
lowercase are whitespace, whether the text must take one of the two other ways below, and, for the character and for
its lowercase, the canonical combining class, the NFC_Quick_Check value and whether it has a canonical decomposition.
One pass over a text deletes, lowercases and collapses the whitespace, and runs the NFC quick check of Unicode
Standard Annex #15 on every word of the result. A character whose quick check is MAYBE passes when canonical
composition cannot join it to the character before it: that character is a starter that has no canonical
decomposition (or, for a starter MAYBE, any starter), and the two are not a composition pair. Only a word that fails
is put into NFC, by unicodedata.normalize. Spaces separate the words, and no canonical composition or reordering
crosses a space. A text of ASCII characters takes a shorter pass with a table of the 128 ASCII characters, because an
ASCII text is in NFC and stays in NFC.

Two kinds of text leave that pass:

- a text with a character whose lowercase depends on the characters around it (the Greek capital sigma) or is
  longer than one character: its punctuation is marked with NUL, the text is lowercased by str.lower, and the same
  pass deletes the NUL characters, collapses the whitespace and puts the words into NFC;
- a text with a character whose canonical equivalents are deleted differently, are whitespace differently or
  lowercase differently: the exact function of werpy._normalize applies the rule step by step.

- normalize_text(text) -> str: one text.
- normalize_texts(texts, element_error) -> list: every text of a list, tuple or one-dimensional numpy array, in
  order. An element that is not a str raises the exception that element_error("text", element, index) returns.
- setup(block, exact, ascii_table): called once by werpy._normalize, with the function that returns the table
  entries of a block, the exact function, and the table of the 128 ASCII characters.
- set_pairs(pairs): called by werpy._normalize with the sorted keys of every primary composite's two characters.
- canonical_decompositions(decomposition, normalize) -> dict: the canonical decomposition mapping of every code point
  that has one, apart from the Hangul syllables. werpy._normalize builds its composition data from it.

Input validation is performed by werpy._normalize.normalize before normalize_text and normalize_texts are called.
"""

from unicodedata import normalize as _unicode_normalize

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.string cimport memcpy

cdef extern from "Python.h":
    const int PY_BIG_ENDIAN
    Py_ssize_t PyUnicode_GetLength(object unicode) except -1
    Py_UCS4* PyUnicode_AsUCS4(object unicode, Py_UCS4* buffer, Py_ssize_t buflen, int copy_null) except NULL
    object PyUnicode_DecodeUTF32(const char* data, Py_ssize_t size, const char* errors, int* byteorder)
    object PyUnicode_DecodeASCII(const char* data, Py_ssize_t size, const char* errors)
    const char* PyBytes_AsString(object o) except NULL
    Py_ssize_t PyBytes_Size(object o) except -1


# One table entry per code point, as werpy._normalize packs it with struct format "=IBBBB".
ctypedef struct Entry:
    uint32_t lower      # the lowercase, one code point
    uint8_t flags       # DELETED, SPACE, SELF_SPACE, SPECIAL, EXACT
    uint8_t ccc         # the canonical combining class of the lowercase
    uint8_t self_ccc    # the canonical combining class of the character
    uint8_t qc          # NFC_Quick_Check of the lowercase in bits 0-1 and of the character in bits 2-3; bit 4 set
                        # when the lowercase has a canonical decomposition, bit 5 when the character has one

cdef enum:
    DELETED = 1         # NUL, an ASCII punctuation character or a character of general category P
    SPACE = 2           # the lowercase is whitespace
    SELF_SPACE = 4      # the character is whitespace
    SPECIAL = 8         # the lowercase depends on the characters around it, or is longer than one character
    EXACT = 16          # the text needs the exact function
    QC_YES = 0
    QC_MAYBE = 1
    QC_NO = 2
    DECOMPOSES = 16     # the lowercase has a canonical decomposition (bit 5, 32, for the character)
    BLOCKS = 0x1100     # blocks of 256 code points up to U+10FFFF
    STACK = 256         # texts up to this length use buffers on the stack
    RESULT_SPECIAL = -1
    RESULT_EXACT = -3

cdef const Entry* _blocks[BLOCKS]
cdef const uint64_t* _pairs = NULL
cdef Py_ssize_t _npairs = 0
cdef list _kept = []
cdef object _block = None
cdef object _exact = None
cdef uint8_t _ascii[128]
cdef int _byteorder = 1 if PY_BIG_ENDIAN else -1


cdef inline object _text_of(const uint32_t* codes, Py_ssize_t n):
    """Return the str of n code points. surrogatepass keeps a lone surrogate."""
    cdef int byteorder = _byteorder
    return PyUnicode_DecodeUTF32(<const char*> codes, n * 4, b"surrogatepass", &byteorder)


@cython.cdivision(True)
cdef inline bint _composes(uint32_t first, uint32_t second) noexcept:
    """
    Return True when canonical composition joins first and second into one character: a Hangul leading consonant and
    medial vowel, or an LV syllable and a final consonant (The Unicode Standard, section 3.12), or a primary composite.
    """
    cdef Py_ssize_t low = 0, high = _npairs, middle
    cdef uint64_t key = (<uint64_t> first << 21) | second
    if 0x1161 <= second <= 0x1175 and 0x1100 <= first <= 0x1112:
        return True
    if 0x11A8 <= second <= 0x11C2 and 0xAC00 <= first <= 0xD7A3 and (first - 0xAC00) % 28 == 0:
        return True
    while low < high:
        middle = (low + high) // 2
        if _pairs[middle] < key:
            low = middle + 1
        else:
            high = middle
    return low < _npairs and _pairs[low] == key


cdef const Entry* _load(Py_ssize_t b) except NULL:
    """Build and return the entries of block b. The callers call it only when no text has needed the block yet."""
    cdef bytes data = _block(b)
    if PyBytes_Size(data) != 256 * sizeof(Entry):
        raise ValueError("A block of the normalization table must hold 256 entries.")
    cdef const Entry* entries = _blocks[b]
    if entries == NULL:
        # Another thread can build the same block while _block runs. The first entries stored are kept.
        _kept.append(data)
        entries = <const Entry*> PyBytes_AsString(data)
        _blocks[b] = entries
    return entries


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _ascii_text(const uint32_t* text, Py_ssize_t n, uint8_t* out):
    """Normalize a text of ASCII characters into out. An ASCII text is in NFC, and stays in NFC."""
    cdef Py_ssize_t i, k = 0
    cdef bint pending = False
    cdef uint8_t value
    for i in range(n):
        value = _ascii[text[i]]
        if value == 0:
            continue
        if value == 0x20:
            if k > 0:
                pending = True
            continue
        if pending:
            out[k] = 0x20
            k += 1
            pending = False
        out[k] = value
        k += 1
    return PyUnicode_DecodeASCII(<const char*> out, k, NULL)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t _fold(const uint32_t* text, Py_ssize_t n, uint32_t* out, Py_ssize_t* words, Py_ssize_t* nwords,
                      bint lowered) except -2:
    """
    Delete, lowercase and collapse the whitespace of text into out, and return the length of the result. The start
    and the end of every word whose NFC quick check is not YES go to words, and their number to nwords.

    When lowered is set, the text is already marked and lowercased: NUL marks a deleted character, and every other
    character is kept as it is. Otherwise the table gives the deletion and the lowercase, and the function returns
    RESULT_SPECIAL or RESULT_EXACT for a text that leaves this pass.
    """
    cdef Py_ssize_t i, k = 0, start = 0, w = 0
    cdef bint pending = False, check = False
    cdef uint8_t last = 0, ccc, qc, flags, decomposes, previous_decomposes = 0
    cdef uint32_t c, o, previous = 0
    cdef const Entry* entries
    cdef const Entry* e
    for i in range(n):
        c = text[i]
        entries = _blocks[c >> 8]
        if entries == NULL:
            entries = _load(c >> 8)
        e = entries + (c & 0xFF)
        flags = e.flags
        if lowered:
            if c == 0:
                continue
            if flags & SELF_SPACE:
                if k > 0:
                    pending = True
                continue
            o = c
            ccc = e.self_ccc
            qc = (e.qc >> 2) & 3
            decomposes = e.qc & (2 * DECOMPOSES)
        else:
            if flags & (SPECIAL | EXACT):
                return RESULT_EXACT if flags & EXACT else RESULT_SPECIAL
            if flags & DELETED:
                continue
            if flags & SPACE:
                if k > 0:
                    pending = True
                continue
            o = e.lower
            ccc = e.ccc
            qc = e.qc & 3
            decomposes = e.qc & DECOMPOSES
        if pending:
            if check:
                words[w] = start
                words[w + 1] = k
                w += 2
            out[k] = 0x20
            k += 1
            start = k
            last = 0
            check = False
            pending = False
            previous = 0
        # The quick check of UAX #15: a mark out of canonical order, or a character that is NFC_Quick_Check NO.
        if (ccc != 0 and last > ccc) or qc == QC_NO:
            check = True
        elif qc == QC_MAYBE and previous != 0:
            # Composition can join a MAYBE character only to the character just before it: a starter MAYBE when
            # that character is a starter, a mark when it is a starter that has no canonical decomposition (so no
            # mark of its own can be reordered after the mark). A MAYBE character at the start of a word follows
            # whitespace, which composes with nothing.
            if ccc == 0:
                if last == 0 and _composes(previous, o):
                    check = True
            elif last != 0 or previous_decomposes or _composes(previous, o):
                check = True
        last = ccc
        previous = o
        previous_decomposes = decomposes
        out[k] = o
        k += 1
    if check:
        words[w] = start
        words[w + 1] = k
        w += 2
    nwords[0] = w // 2
    return k


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _finish(const uint32_t* out, Py_ssize_t k, const Py_ssize_t* words, Py_ssize_t nwords):
    """Return the text of out, with every word that failed the quick check put into NFC."""
    if nwords == 0:
        return _text_of(out, k)
    cdef Py_ssize_t j, start, end, length, done = 0, r = 0, checked = 0
    for j in range(nwords):
        checked += words[2 * j + 1] - words[2 * j]
    if k - checked < 24 * (nwords - 1):
        # The factor in the condition above is the cost of one call, counted in characters put into NFC. One call for the
        # whole text costs less than a call for each word when the words that pass are shorter than that factor.
        return _unicode_normalize("NFC", _text_of(out, k))
    # NFC never makes a text more than three times longer.
    cdef uint32_t* result = <uint32_t*> PyMem_Malloc((k + 2 * checked + 1) * sizeof(uint32_t))
    if result == NULL:
        raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
    try:
        for j in range(nwords):
            start = words[2 * j]
            end = words[2 * j + 1]
            memcpy(result + r, out + done, (start - done) * sizeof(uint32_t))
            r += start - done
            word = _unicode_normalize("NFC", _text_of(out + start, end - start))
            length = PyUnicode_GetLength(word)
            PyUnicode_AsUCS4(word, <Py_UCS4*> (result + r), length, 0)
            r += length
            done = end
        memcpy(result + r, out + done, (k - done) * sizeof(uint32_t))
        r += k - done
        normalized = _text_of(result, r)
    finally:
        PyMem_Free(result)
    return normalized


cdef object _special(object text, uint32_t* codes, Py_ssize_t n):
    """Mark the deleted characters with NUL, lowercase the text with str.lower, then finish it in the pass."""
    cdef Py_ssize_t i, m, k, nwords = 0
    cdef uint32_t c
    cdef const Entry* entries
    for i in range(n):
        c = codes[i]
        entries = _blocks[c >> 8]
        if entries == NULL:
            entries = _load(c >> 8)
        if entries[c & 0xFF].flags & EXACT:
            return _exact(text)
        if entries[c & 0xFF].flags & DELETED:
            codes[i] = 0
    lowered = _text_of(codes, n).lower()
    m = PyUnicode_GetLength(lowered)
    cdef void* memory = PyMem_Malloc((m + 2) * sizeof(Py_ssize_t) + 2 * m * sizeof(uint32_t))
    if memory == NULL:
        raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
    cdef Py_ssize_t* words = <Py_ssize_t*> memory
    cdef uint32_t* lowered_codes = <uint32_t*> (words + m + 2)
    cdef uint32_t* out = lowered_codes + m
    try:
        PyUnicode_AsUCS4(lowered, <Py_UCS4*> lowered_codes, m, 0)
        k = _fold(lowered_codes, m, out, words, &nwords, True)
        normalized = _finish(out, k, words, nwords)
    finally:
        PyMem_Free(memory)
    return normalized


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _normalize_one(object text):
    """Return the normalized form of one str."""
    cdef Py_ssize_t n = PyUnicode_GetLength(text)
    if n == 0:
        return ""
    cdef Py_ssize_t i, k, nwords = 0
    cdef uint32_t bits = 0
    cdef Py_ssize_t stack_words[STACK + 2]
    cdef uint32_t stack_codes[STACK]
    cdef uint32_t stack_out[STACK]
    cdef Py_ssize_t* words = stack_words
    cdef uint32_t* codes = stack_codes
    cdef uint32_t* out = stack_out
    cdef void* memory = NULL
    if n > STACK:
        memory = PyMem_Malloc((n + 2) * sizeof(Py_ssize_t) + 2 * n * sizeof(uint32_t))
        if memory == NULL:
            raise MemoryError()  # pragma: no cover - only an allocator failure reaches this line
        words = <Py_ssize_t*> memory
        codes = <uint32_t*> (words + n + 2)
        out = codes + n
    try:
        PyUnicode_AsUCS4(text, <Py_UCS4*> codes, n, 0)
        for i in range(n):
            bits |= codes[i]
        if bits < 0x80:
            normalized = _ascii_text(codes, n, <uint8_t*> out)
        else:
            k = _fold(codes, n, out, words, &nwords, False)
            if k == RESULT_EXACT:
                normalized = _exact(text)
            elif k == RESULT_SPECIAL:
                normalized = _special(text, codes, n)
            else:
                normalized = _finish(out, k, words, nwords)
    finally:
        PyMem_Free(memory)
    return normalized


def normalize_text(object text):
    """Return the normalized form of one str, of a str subclass or of numpy.str_."""
    return _normalize_one(text)


def normalize_texts(object texts, object element_error):
    """Return the list of the normalized forms of the elements of texts."""
    cdef list result = []
    cdef Py_ssize_t index = 0
    for text in texts:
        if not isinstance(text, str):
            raise element_error("text", text, index)
        result.append(_normalize_one(text))
        index += 1
    return result


def setup(object block, object exact, bytes ascii_table not None):
    """Store the function that returns the entries of a block, the exact function and the ASCII table."""
    global _block, _exact
    if len(ascii_table) != 128:
        raise ValueError("The ASCII table must hold 128 entries.")
    memcpy(_ascii, PyBytes_AsString(ascii_table), 128)
    _block = block
    _exact = exact


def set_pairs(bytes pairs not None):
    """
    Store the sorted keys (first << 21 | second, as 8-byte unsigned integers) of the two characters of every primary
    composite. The keys stored before stay alive, as a pass on another thread may still read them.
    """
    global _pairs, _npairs
    if len(pairs) % 8:
        raise ValueError("The composition pairs must be 8-byte keys.")
    _kept.append(pairs)
    _pairs = <const uint64_t*> PyBytes_AsString(pairs)
    _npairs = len(pairs) // 8


def canonical_decompositions(object decomposition, object normalize):
    """
    Return the canonical decomposition mapping of every code point that has one, as a dict from the code point to the
    string that decomposition() returns. A block of 256 code points is read one code point at a time only when NFD
    changes the string of its code points, since NFD replaces every character that has a canonical decomposition. The
    Hangul syllables U+AC00 to U+D7A3 are left out: they compose by the algorithm of The Unicode Standard, section 3.12.
    """
    cdef uint32_t codes[4096]
    cdef dict result = {}
    cdef Py_ssize_t chunk, block, i, cp
    for chunk in range(BLOCKS // 16):
        for i in range(4096):
            codes[i] = <uint32_t> (chunk * 4096 + i)
        text = _text_of(codes, 4096)
        if normalize("NFD", text) == text:
            continue
        for block in range(16):
            text = _text_of(codes + block * 256, 256)
            if normalize("NFD", text) == text:
                continue
            for i in range(256):
                cp = chunk * 4096 + block * 256 + i
                if 0xAC00 <= cp <= 0xD7A3:
                    continue
                mapping = decomposition(chr(<int> cp))
                if mapping and not mapping.startswith("<"):
                    result[cp] = mapping
    return result
