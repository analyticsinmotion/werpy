# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
normalize_reference.py

This module is a pure-Python reference for the rule of the 'normalize' function of the 'werpy' package. The tests in
'test_normalize_rules.py' compare 'normalize' with it. It imports nothing from 'werpy'.

Each function is written from the rule, not from the code of werpy: one character at a time, with no regular
expression, no table, no ASCII shortcut and no cache. The Unicode primitives are the ones that the rule names, all from
the Unicode database of the running Python: unicodedata.normalize for NFC, unicodedata.category for the general
category, str.lower for the lowercase, which applies the final sigma of the Greek capital sigma, and str.isspace for
whitespace.

A removed character is first replaced by NUL, the text is converted to lowercase, and the NUL characters are then
removed. NUL is neither cased nor case-ignorable, so a removed character still ends a word for the final sigma.

    n1          the rule of normalize: NFC; NUL, the ASCII punctuation characters and every character of the general
                category P removed; lowercase; NFC; whitespace collapsed. Lone surrogates are kept.
    today_keep  the rule of werpy 3.5.0 with lone surrogates kept: NUL and the ASCII punctuation characters removed;
                lowercase; whitespace collapsed.
"""

import string
import unicodedata

NUL = chr(0)
SPACE = chr(0x20)
ASCII_PUNCTUATION = frozenset(string.punctuation)


def collapse_whitespace(text):
    """Split at every run of characters for which str.isspace() is true, drop empty parts, join with one space."""
    words = []
    current = []
    for c in text:
        if c.isspace():
            if current:
                words.append("".join(current))
                current = []
        else:
            current.append(c)
    if current:
        words.append("".join(current))
    return SPACE.join(words)


def is_punctuation(c):
    """Return True for NUL, one of the ASCII punctuation characters, or a character of the general category P."""
    return c == NUL or c in ASCII_PUNCTUATION or unicodedata.category(c).startswith("P")


def delete_marked(text, marked):
    """Replace every character for which marked(c) is true by NUL, convert to lowercase, then remove the NULs."""
    placeholders = "".join(NUL if marked(c) else c for c in text)
    return placeholders.lower().replace(NUL, "")


def today_keep(text):
    """Return the result of the rule of werpy 3.5.0, with lone surrogates kept."""
    return collapse_whitespace(delete_marked(text, lambda c: c == NUL or c in ASCII_PUNCTUATION))


def n1(text):
    """Return the result of the rule of normalize."""
    text = unicodedata.normalize("NFC", text)
    text = delete_marked(text, is_punctuation)
    return collapse_whitespace(unicodedata.normalize("NFC", text))
