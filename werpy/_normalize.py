# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
The _normalize module provides the normalize function, which prepares texts for the Word Error Rate (WER)
functions. It puts each text into Unicode Normalization Form C (NFC), removes the punctuation, converts the text to
lowercase and collapses the whitespace. The compiled module werpy._normalize_kernel applies this rule to the characters
of a text. This module checks the input and builds the tables that the kernel reads from the Unicode database of the
running Python.

This module defines the following function:
    - normalize(text)
"""

from __future__ import annotations

import string
import struct
import unicodedata
from functools import cache
from typing import TYPE_CHECKING, overload

from ._errorhandler import check_text_input, element_error
from ._normalize_kernel import canonical_decompositions, normalize_text, normalize_texts, set_pairs, setup

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


# The table that werpy._normalize_kernel reads holds one entry per code point: the lowercase, the flags, the
# canonical combining classes of the lowercase and of the character, and their NFC_Quick_Check values (0 YES, 1 MAYBE,
# 2 NO) with a bit for each that has a canonical decomposition. It is built from the Unicode database of the running
# Python, one block of 256 code points at a time, when a text first holds a character of the block.
_ENTRY = struct.Struct("=IBBBB")
_DELETED, _SPACE, _SELF_SPACE, _SPECIAL, _EXACT = 1, 2, 4, 8, 16
_YES, _MAYBE, _NO = 0, 1, 2

# The medial vowels and the final consonants of the Hangul syllables. They compose by the algorithm of The Unicode
# Standard, section 3.12, and not through decomposition mappings.
_HANGUL_SECONDS = frozenset(range(0x1161, 0x1176)) | frozenset(range(0x11A8, 0x11C3))


def _is_deleted(c: str) -> bool:
    """Return True for NUL, the ASCII punctuation characters and every character of Unicode general category P."""
    return c == "\x00" or c in string.punctuation or unicodedata.category(c).startswith("P")


def _consistent(c: str, equivalent: str) -> bool:
    """
    Return True if a character and a canonically equivalent sequence give the same result: the rule deletes all or
    none of them, all or none of them are whitespace, and their lowercases are canonically equivalent.
    """
    chars = c + equivalent
    return (
        len({_is_deleted(x) for x in chars}) == 1
        and len({x.isspace() for x in chars}) == 1
        and unicodedata.normalize("NFC", c.lower()) == unicodedata.normalize("NFC", equivalent.lower())
    )


@cache
def _composition() -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    """
    Return, from the Unicode database of the running Python, the characters that canonical composition joins to the
    character before them (NFC_Quick_Check MAYBE), the characters it joins to the character after them, and the
    characters for which the kernel uses the exact function. The two characters of every primary composite go to the
    kernel as sorted keys.

    The kernel deletes and lowercases a text as it is given, and puts only the result into NFC. That gives the rule's
    result unless the text holds a character that NFC changes, or the second character of a composition, whose
    canonical equivalents differ in deletion, in whitespace or in their lowercase.
    """
    seconds: set[int] = set(_HANGUL_SECONDS)
    firsts: set[int] = set()
    exact: set[int] = set()
    pairs: list[int] = []
    for cp, mapping in canonical_decompositions(unicodedata.decomposition, unicodedata.normalize).items():
        c = chr(cp)
        composed = unicodedata.normalize("NFC", c)
        if composed != c:
            if not _consistent(c, composed):
                exact.add(cp)
            continue
        parts = [chr(int(code, 16)) for code in mapping.split()]
        if len(parts) == 2:
            firsts.add(ord(parts[0]))
            seconds.add(ord(parts[1]))
            pairs.append(ord(parts[0]) << 21 | ord(parts[1]))
            if not _consistent(c, parts[0] + parts[1]):
                exact.add(ord(parts[1]))
    set_pairs(struct.pack(f"={len(pairs)}Q", *sorted(pairs)))
    return frozenset(seconds), frozenset(firsts), frozenset(exact)


def _decomposes(c: str) -> int:
    """Return 1 when the character has a canonical decomposition, 0 otherwise."""
    return int(unicodedata.normalize("NFD", c) != c)


def _quick_check(c: str, seconds: frozenset[int]) -> int:
    """
    Return the NFC_Quick_Check value of one character: NO when NFC changes it, MAYBE when composition can join it to
    the character before it, YES otherwise.
    """
    if unicodedata.normalize("NFC", c) != c:
        return _NO
    return _MAYBE if ord(c) in seconds else _YES


def _needs_exact(c: str, deleted: bool, seconds: frozenset[int], firsts: frozenset[int], exact: frozenset[int]) -> bool:
    """
    Return True when a text that holds the character needs the exact function: the character is in the exact set, or
    it is a deleted mark, which could move in canonical reordering, or it is whitespace that composition or
    reordering could cross, so that it would not separate words.
    """
    cp = ord(c)
    deleted_mark = deleted and unicodedata.combining(c) != 0
    crossed_space = c.isspace() and (cp in seconds or cp in firsts or unicodedata.combining(c) != 0)
    return cp in exact or deleted_mark or crossed_space


def _block(block: int) -> bytes:
    """Return the table entries of the 256 code points of one block."""
    seconds, firsts, exact = _composition()
    entries = bytearray(256 * _ENTRY.size)
    for offset in range(256):
        c = chr(block * 256 + offset)
        lower = c.lower()
        flags = _DELETED if _is_deleted(c) else 0
        if c.isspace():
            flags |= _SELF_SPACE
        if len(lower) != 1 or ("A" + c).lower()[1:] != lower or (c + "A").lower()[:-1] != lower:
            # The lowercase is longer than one character, or depends on the characters around it (final sigma).
            flags |= _SPECIAL
            lower = c
        if lower.isspace():
            flags |= _SPACE
        if _needs_exact(c, bool(flags & _DELETED), seconds, firsts, exact):
            flags |= _EXACT
        _ENTRY.pack_into(
            entries,
            offset * _ENTRY.size,
            ord(lower),
            flags,
            unicodedata.combining(lower),
            unicodedata.combining(c),
            _quick_check(lower, seconds)
            | _quick_check(c, seconds) << 2
            | _decomposes(lower) << 4
            | _decomposes(c) << 5,
        )
    return bytes(entries)


def _ascii_table() -> bytes:
    """
    Return the table of the 128 ASCII characters for the kernel: 0 for a deleted character, 32 for whitespace, and
    the lowercase for every other character.
    """
    return bytes(0 if _is_deleted(c) else 32 if c.isspace() else ord(c.lower()) for c in map(chr, range(128)))


def _normalize_exact(text: str) -> str:
    """
    Apply the rule step by step to a text that holds a character of the exact set: NFC, deleted characters marked
    with NUL, lowercase, NUL deleted, NFC, whitespace collapsed.
    """
    text = unicodedata.normalize("NFC", text)
    marked = "".join("\x00" if _is_deleted(c) else c for c in text)
    return " ".join(unicodedata.normalize("NFC", marked.lower().replace("\x00", "")).split())


setup(_block, _normalize_exact, _ascii_table())


@overload
def normalize(text: str) -> str: ...


@overload
def normalize(
    text: list[str] | tuple[str, ...] | NDArray[np.str_] | NDArray[np.object_],
) -> list[str]: ...


def normalize(
    text: str | list[str] | tuple[str, ...] | NDArray[np.str_] | NDArray[np.object_],
) -> str | list[str]:
    """
    This function prepares texts for the Word Error Rate (WER) functions and for other natural language processing
    tasks. It applies the same rule to each text, in this order:

    1. The text is put into Unicode Normalization Form C (NFC). A character that is written as one code point and
       the same character written as a letter followed by combining marks become the same text.
    2. NUL, the ASCII punctuation characters of string.punctuation and every character of the Unicode general
       category for punctuation are removed. The characters on each side of a removed character join, so
       well-known becomes wellknown.
    3. The text is converted to lowercase. A removed punctuation character still ends a word for the Greek capital
       sigma, which becomes the final sigma before it.
    4. The text is put into NFC again, because a removed character or the lowercase can let characters compose.
    5. Every run of whitespace becomes one space, and the whitespace at the start and at the end is removed.

    What happens to each kind of character:

    +--------------------------------------------+------------------------------------------------------------------+
    | Character                                  | Result                                                           |
    +============================================+==================================================================+
    | Letters, and the other characters that     | Converted to lowercase.                                          |
    | have a lowercase form, such as the Roman   |                                                                  |
    | numerals                                   |                                                                  |
    +--------------------------------------------+------------------------------------------------------------------+
    | Letters with accents                       | Put into the composed form, so a letter with an accent as one    |
    |                                            | character and the letter followed by a combining accent give the |
    |                                            | same result.                                                     |
    +--------------------------------------------+------------------------------------------------------------------+
    | The ASCII punctuation characters of        | Removed. The characters on each side join, so well-known becomes |
    | string.punctuation                         | wellknown.                                                       |
    +--------------------------------------------+------------------------------------------------------------------+
    | Other punctuation, such as the curly       | Removed in the same way.                                         |
    | quotation marks, the dashes, the ellipsis  |                                                                  |
    | and the ideographic full stop              |                                                                  |
    +--------------------------------------------+------------------------------------------------------------------+
    | Symbols other than ASCII, such as the euro | Kept.                                                            |
    | sign, and emoji                            |                                                                  |
    +--------------------------------------------+------------------------------------------------------------------+
    | Digits and numbers                         | Kept.                                                            |
    +--------------------------------------------+------------------------------------------------------------------+
    | Spaces, tabs, line breaks and the other    | Each run becomes one space. The whitespace at the start and at   |
    | characters for which str.isspace() is true | the end is removed.                                              |
    +--------------------------------------------+------------------------------------------------------------------+
    | Invisible format characters, such as the   | Kept.                                                            |
    | zero width space and the soft hyphen       |                                                                  |
    +--------------------------------------------+------------------------------------------------------------------+
    | NUL                                        | Removed.                                                         |
    +--------------------------------------------+------------------------------------------------------------------+
    | Lone surrogates                            | Kept.                                                            |
    +--------------------------------------------+------------------------------------------------------------------+
    | Every other character, such as the other   | Kept.                                                            |
    | control characters and the private use     |                                                                  |
    | characters                                 |                                                                  |
    +--------------------------------------------+------------------------------------------------------------------+

    normalize is idempotent: a result that is normalized again does not change.

    The rule uses the Unicode database of the Python that runs it. A text whose characters are all assigned in the
    Unicode version of the oldest Python that werpy supports gives the same result on every supported Python. A
    character that only a later version of Unicode assigns, such as a new punctuation character or a new combining
    mark, can give a different result on an older Python.

    Parameters
    ----------
    text : str, list, tuple or numpy array
        The input text to be normalized.

    Raises
    ------
    werpy.exceptions.WerpyTypeError
        If the input is not a string, a list, a tuple or a numpy array, if a numpy
        array does not have exactly one dimension, or if an element of a list, tuple
        or numpy array is not a string. Other iterables, for example a pandas Series,
        a set or a generator, raise this exception.

    Returns
    -------
    str or list
        If the input is a string, the function returns the normalized string. If the
        input is a list, tuple, or numpy array of strings, it returns a list of
        normalized strings.

    Examples
    --------
    >>> reference = normalize(" it's Consumed Domestically  And exported to other countries.")
    >>> print(reference)
    its consumed domestically and exported to other countries
    >>> reference
    'its consumed domestically and exported to other countries'

    >>> input_data = ["It's very popular in Antarctica.","The Sugar Bear character"]
    >>> reference = normalize(input_data)
    >>> print(reference)
    ['its very popular in antarctica', 'the sugar bear character']
    >>> reference
    ['its very popular in antarctica', 'the sugar bear character']

    >>> normalize("It\N{RIGHT SINGLE QUOTATION MARK}s done \N{EM DASH} cafe\N{COMBINING ACUTE ACCENT}")
    'its done caf\N{LATIN SMALL LETTER E WITH ACUTE}'
    """
    check_text_input("text", text)

    if isinstance(text, str):
        return normalize_text(text)

    return normalize_texts(text, element_error)
