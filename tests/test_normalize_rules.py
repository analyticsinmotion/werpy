# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_normalize_rules.py

This module contains a set of unit tests for the rule of the 'normalize' function of the 'werpy' package. normalize
puts each text into Unicode Normalization Form C (NFC), removes NUL, the ASCII punctuation characters and every
character of the Unicode general category for punctuation, converts the text to lowercase, puts it into NFC again and
collapses the whitespace. Lone surrogates are kept.

The tests compare normalize with the reference 'n1' of 'normalize_reference.py' on every code point of the classes of
characters that the rule treats in different ways and on seeded random texts, and check that normalize is idempotent
and that ASCII texts keep the result of werpy 3.5.0. They check the classes of the documentation, canonically
equivalent texts, lone surrogates, the batch forms, the element error, punctuation above U+FFFF, the final sigma, the
Word Error Rate of normalized texts and the outputs of the example texts.

Other tests reach each way through the compiled module 'werpy._normalize_kernel' with the texts that take it: the
ASCII pass, the pass with the NFC quick check, a text with a character whose lowercase depends on its neighbours, and
a text that the step-by-step function of 'werpy._normalize' handles. They compare the kernel with that function on the
same inputs, check canonical composition after every second character of a composition, check the first texts of a
new Python process, and check the setup functions of the kernel. Texts with characters that some versions of Unicode
do not assign are compared with the reference only, and their outputs are not pinned.

To run the tests, execute this module as the main program.
"""

import ast
import functools
import importlib
import random
import string
import subprocess  # nosec B404
import sys
import unicodedata
import unittest
import normalize_reference as reference
import werpy
from werpy.exceptions import WerpyTypeError, WerpyValueError

try:
    import numpy as np
except ImportError:  # NumPy is not a dependency of werpy
    np = None


NORMALIZE = importlib.import_module("werpy._normalize")
KERNEL = importlib.import_module("werpy._normalize_kernel")
SEED = 20261101
# The general categories whose code points the property tests take one by one: punctuation, symbols, marks,
# separators, control and format characters, surrogates, and uppercase and titlecase letters.
CATEGORIES = ("P", "S", "M", "Z", "Cc", "Cf", "Cs", "Lu", "Lt")
ACUTE, CARON, DOT_BELOW = chr(0x301), chr(0x30C), chr(0x323)
SIGMA_WORD = chr(0x3A3) + chr(0x391) + chr(0x3A3)
FINAL_SIGMA_WORD = chr(0x3C3) + chr(0x3B1) + chr(0x3C2)
EMOJI = chr(0x1F600)


def chars(*parts):
    """Return the text of the parts: a str part as it is, an int part as the character of that code point."""
    return "".join(chr(part) if isinstance(part, int) else part for part in parts)


def numpy_forms(texts):
    """Return the texts as numpy arrays of dtype U and object, or no form when NumPy cannot be imported."""
    return [] if np is None else [np.array(texts, dtype=str), np.array(texts, dtype=object)]


class Text(str):
    """A str subclass."""


@functools.cache
def code_point_scan():
    """
    Return two lists from one scan of every code point: the code points of CATEGORIES or with a canonical
    decomposition, with every 17th Hangul syllable and every syllable without a final consonant; and the primary
    composites as (composite, first, second).
    """
    singles, composites = [], []
    for cp in range(sys.maxunicode + 1):
        c = chr(cp)
        decomposes = unicodedata.normalize("NFD", c) != c
        if 0xAC00 <= cp <= 0xD7A3:
            if (cp - 0xAC00) % 17 == 0 or (cp - 0xAC00) % 28 == 0:
                singles.append(cp)
            continue
        if decomposes or unicodedata.category(c).startswith(CATEGORIES):
            singles.append(cp)
        if decomposes and unicodedata.normalize("NFC", c) == c:
            parts = [int(code, 16) for code in unicodedata.decomposition(c).split()]
            if len(parts) == 2:
                composites.append((cp, parts[0], parts[1]))
    return singles, composites


@functools.cache
def property_texts():
    """
    Return the texts of the property tests: every code point of code_point_scan alone and between two letters, and
    5,000 seeded random texts of 1 to 12 characters drawn from those code points, the ASCII characters and the
    characters whose lowercase or canonical equivalents need care.
    """
    singles, _composites = code_point_scan()
    texts = [chr(cp) for cp in singles] + ["a" + chr(cp) + "b" for cp in singles]
    rng = random.Random(SEED)
    pool = singles + list(range(128)) + [0x130, 0x3A3, 0x338, 0x1FEF, 0x301, 0x323, 0x20, 0x20]
    texts += ["".join(chr(rng.choice(pool)) for _ in range(rng.randint(1, 12))) for _ in range(5000)]
    return texts


def kernel_texts():
    """
    Return (name, text) for each way through the compiled kernel: the empty text; the ASCII pass; the pass with the
    NFC quick check, a deleted character and whitespace; a mark out of canonical order, a character that NFC changes
    and a word that passes; the text with a capital sigma or a dotted capital I, with a deleted character and with a
    character of the exact set inside; the text of the exact set; the final NFC of one word and of the whole text;
    each way with texts just below, at and above the length of the stack buffers.
    """
    cases = [
        ("empty", ""),
        ("ASCII, deleted characters and whitespace at the start, between words and at the end", " a, b "),
        ("quick check pass, deleted characters, leading and repeated whitespace",
         chars("  ", 0xE9, ", ", 0x2014, "  ", 0xFC, " ")),
        ("a mark out of canonical order", chars("a", ACUTE, DOT_BELOW)),
        ("a character that NFC changes", chars(0x212B, "ngstr", 0xF6, "m")),
        ("a word that passes the quick check", chars("caf", 0xE9, " ", 0xE0, " la")),
        ("capital sigma with a deleted character", SIGMA_WORD + ", " + SIGMA_WORD),
        ("capital sigma with whitespace and a mark", chars(" ", SIGMA_WORD, ACUTE, "  ", 0x3A3, " ")),
        ("dotted capital I with a character of the exact set inside", chars(0x130, 0x338, " x")),
        ("capital sigma before a block that no text has used", chars(0x3A3, "A", 0xD0001)),
        ("exact set, not equal to by a combining long solidus overlay", chars("a =", 0x338, " b")),
        ("exact set, Greek varia", chars("x", 0x1FEF, "y")),
        ("final NFC of one word", "word " * 50 + "cafe" + ACUTE),
        ("final NFC of the whole text", ("e" + ACUTE + " ") * 100),
    ]
    for n in (255, 256, 257, 1000):
        cases.append((f"ASCII, {n} characters", ("Ab, " * n)[:n]))
        cases.append((f"quick check pass, {n} characters", (chars("Ab", 0xE9, " ", 0x2014) * n)[:n]))
        cases.append((f"capital sigma, {n} characters", ((SIGMA_WORD + "A, ") * n)[:n]))
        cases.append((f"exact set, {n} characters", (chars("a =", 0x338, " ") * n)[:n]))
        cases.append((f"final NFC of one word, {n} characters", ("word " * n)[: n - 5] + "cafe" + ACUTE))
    return cases


def maybe_texts():
    """
    Return (name, text) for each case of the quick check of a character whose NFC_Quick_Check is MAYBE: a starter
    after a starter that it composes with, after one that it does not compose with, after a mark and at the start of
    a word; a mark after a starter with no canonical decomposition that it does not compose with, after a starter that
    it composes with, after a starter that has a canonical decomposition and after another mark; the Hangul syllables;
    and sentences in Tamil, Bengali and Hindi.
    """
    tamil = chars(0xB95, 0xBBE, 0xBB2, 0xBAE, 0xBCD, " ", 0xBAA, 0xBBE, 0xB9F, 0xBAE, 0xBCD, " ", 0xBA4, 0xBAE,
                  0xBBF, 0xBB4, 0xBCD, " ", 0xBB5, 0xBBE, 0xBB4, 0xBCD, 0xB95, 0xBC8, 0xBAF, 0xBBF, 0xBB2, 0xBCD)
    bengali = chars(0x986, 0x9AE, 0x9BE, 0x9B0, " ", 0x9AD, 0x9BE, 0x9B7, 0x9BE, " ", 0x9AC, 0x9BE, 0x982, 0x9B2,
                    0x9BE, " ", 0x995, 0x9BE, 0x9B2, 0x9CB)
    hindi = chars(0x92A, 0x922, 0x93C, 0x93E, 0x908, " ", 0x92B, 0x93C, 0x93F, 0x932, 0x94D, 0x92E, " ", 0x91C,
                  0x93C, 0x930, 0x942, 0x930, 0x924, " ", 0x921, 0x93C, 0x940)
    return [
        ("a starter after a starter that it composes with", chars(0xBC6, 0xBBE)),
        ("a starter after a starter that it does not compose with", chars(0xB95, 0xBBE)),
        ("a starter after a mark", chars(ACUTE, 0xBBE)),
        ("a starter at the start of a word", chars("a ", 0xBBE, " ", 0x9BE)),
        ("a mark after a starter with no decomposition that it does not compose with", chars(0x921, 0x93C)),
        ("a mark after a starter that it composes with", chars("e", ACUTE, " A", 0x30A)),
        ("a mark after a starter that has a canonical decomposition", chars(0xE9, DOT_BELOW)),
        ("a mark after another mark", chars("a", DOT_BELOW, ACUTE)),
        ("a Hangul leading consonant and vowel", chars(0x1100, 0x1161, " ", 0x1112, 0x1161, 0x11AB)),
        ("a Hangul syllable without a final consonant, and a final consonant", chars(0xAC00, 0x11A8)),
        ("a Hangul syllable with a final consonant, and a final consonant", chars(0xAC01, 0x11A8)),
        ("Tamil", tamil),
        ("Bengali", bengali),
        ("Hindi", hindi),
        ("Tamil, Bengali and Hindi in decomposed form",
         unicodedata.normalize("NFD", tamil + " " + bengali + " " + hindi)),
    ]


# The rows of the table of the docstring of normalize and of the normalization page: (row, text, result).
DOCUMENTED_CLASSES = (
    ("Letters, and the other characters that have a lowercase form",
     chars("ABC D", 0xC9, "f ", 0x216B, " ", 0x24B6), chars("abc d", 0xE9, "f ", 0x217B, " ", 0x24D0)),
    ("Letters with accents", chars("cafe", ACUTE, " CAF", 0xC9), chars("caf", 0xE9, " caf", 0xE9)),
    ("The ASCII punctuation characters", "well-known, it's 3.14 $5 a@b.com (c) <d>", "wellknown its 314 5 abcom c d"),
    ("Other punctuation",
     chars("It", 0x2019, "s ", 0x201C, "done", 0x201D, " ", 0x2014, " ", 0x2026, " ", 0xAB, "oui", 0xBB, " ", 0xBF,
           "qu", 0xE9, "? ", 0x4F60, 0x597D, 0xFF0C, 0x4E16, 0x754C, 0x3002),
     chars("its done oui qu", 0xE9, " ", 0x4F60, 0x597D, 0x4E16, 0x754C)),
    ("Symbols other than ASCII, and emoji",
     chars(0x20AC, "5 ", 0xA9, " ", 0x2122, " 25", 0xB0, "C ", EMOJI), chars(0x20AC, "5 ", 0xA9, " ", 0x2122, " 25",
                                                                            0xB0, "c ", EMOJI)),
    ("Digits and numbers",
     chars(0x663, 0x664, 0x665, " ", 0xFF13, 0xFF14, " x", 0xB2, " ", 0xBD, " ", 0x2460),
     chars(0x663, 0x664, 0x665, " ", 0xFF13, 0xFF14, " x", 0xB2, " ", 0xBD, " ", 0x2460)),
    ("Spaces, tabs, line breaks and other whitespace",
     chars("  a\tb\nc", 0xA0, "d", 0x3000, "e", 0x2028, "f\x1fg  "), "a b c d e f g"),
    ("Invisible format characters",
     chars("a", 0x200B, "b co", 0xAD, "operate"), chars("a", 0x200B, "b co", 0xAD, "operate")),
    ("NUL", "a\x00b", "ab"),
    ("Lone surrogates", chars("a", 0xD800, "b ", 0xDC80), chars("a", 0xD800, "b ", 0xDC80)),
    ("Every other character", chars("a\x07b ", 0xE000), chars("a\x07b ", 0xE000)),
)

# The example texts of the normalisation design, with the results of the reference, which do not depend on the
# version of Unicode: (name, text, result).
PINNED_EXAMPLES = (
    ("ascii-sentence",
     " it's Consumed Domestically  And exported to other countries.",
     "its consumed domestically and exported to other countries"),
    ("ascii-hyphen",
     "well-known",
     "wellknown"),
    ("ascii-comma-join",
     "hello,world",
     "helloworld"),
    ("ascii-numbers",
     "3.14 $5 1,000 50%",
     "314 5 1000 50"),
    ("ascii-email",
     "e-mail: a@b.com",
     "email abcom"),
    ("ascii-brackets",
     "(a) [b] {c} <d>",
     "a b c d"),
    ("ascii-symbols",
     "a~b|c^d`e+f=g",
     "abcdefg"),
    ("ascii-underscore",
     "snake_case #tag",
     "snakecase tag"),
    ("nul",
     chars("a", 0x0, "b"),
     "ab"),
    ("bell",
     chars("a", 0x7, "b"),
     chars("a", 0x7, "b")),
    ("del",
     chars("a", 0x7F, "b"),
     chars("a", 0x7F, "b")),
    ("unit-separator",
     chars("a", 0x1F, "b"),
     "a b"),
    ("next-line",
     chars("a", 0x85, "b"),
     "a b"),
    ("c1-control",
     chars("a", 0x80, "b"),
     chars("a", 0x80, "b")),
    ("cafe-nfc",
     chars("caf", 0xE9),
     chars("caf", 0xE9)),
    ("cafe-nfd",
     chars("cafe", 0x301),
     chars("caf", 0xE9)),
    ("cafe-upper-nfc",
     chars("CAF", 0xC9),
     chars("caf", 0xE9)),
    ("cafe-upper-nfd",
     chars("CAFE", 0x301),
     chars("caf", 0xE9)),
    ("angstrom-sign",
     chars(0x212B, "ngstr", 0xF6, "m"),
     chars(0xE5, "ngstr", 0xF6, "m")),
    ("vietnamese-nfc",
     chars("Ti", 0x1EBF, "ng Vi", 0x1EC7, "t"),
     chars("ti", 0x1EBF, "ng vi", 0x1EC7, "t")),
    ("vietnamese-nfd",
     chars("Tie", 0x302, 0x301, "ng Vie", 0x323, 0x302, "t"),
     chars("ti", 0x1EBF, "ng vi", 0x1EC7, "t")),
    ("hangul-nfc",
     chars(0xD55C, 0xAD6D, 0xC5B4),
     chars(0xD55C, 0xAD6D, 0xC5B4)),
    ("hangul-nfd",
     chars(0x1112, 0x1161, 0x11AB, 0x1100, 0x116E, 0x11A8, 0x110B, 0x1165),
     chars(0xD55C, 0xAD6D, 0xC5B4)),
    ("j-caron-upper",
     chars("J", 0x30C),
     chars(0x1F0)),
    ("i-dot-upper",
     chars(0x130, "stanbul"),
     chars("i", 0x307, "stanbul")),
    ("greek-final-sigma",
     chars(0x39F, 0x394, 0x3A5, 0x3A3, 0x3A3, 0x395, 0x3A5, 0x3A3),
     chars(0x3BF, 0x3B4, 0x3C5, 0x3C3, 0x3C3, 0x3B5, 0x3C5, 0x3C2)),
    ("greek-question",
     chars(0x3A4, 0x3B9, 0x37E),
     chars(0x3C4, 0x3B9)),
    ("greek-ano-teleia",
     chars(0x3B1, 0x387, 0x3B2),
     chars(0x3B1, 0x3B2)),
    ("cyrillic",
     chars(0x41F, 0x440, 0x438, 0x432, 0x435, 0x442, ", ", 0x43C, 0x438, 0x440, "!"),
     chars(0x43F, 0x440, 0x438, 0x432, 0x435, 0x442, " ", 0x43C, 0x438, 0x440)),
    ("arabic",
     chars(0x645, 0x631, 0x62D, 0x628, 0x627, 0x60C, " ", 0x639, 0x627, 0x644, 0x645, 0x61F),
     chars(0x645, 0x631, 0x62D, 0x628, 0x627, " ", 0x639, 0x627, 0x644, 0x645)),
    ("hebrew-maqaf",
     chars(0x5D1, 0x5D9, 0x5EA, 0x5BE, 0x5E1, 0x5E4, 0x5E8),
     chars(0x5D1, 0x5D9, 0x5EA, 0x5E1, 0x5E4, 0x5E8)),
    ("devanagari-danda",
     chars(0x928, 0x92E, 0x938, 0x94D, 0x924, 0x947, " ", 0x926, 0x941, 0x928, 0x93F, 0x92F, 0x93E, 0x964),
     chars(0x928, 0x92E, 0x938, 0x94D, 0x924, 0x947, " ", 0x926, 0x941, 0x928, 0x93F, 0x92F, 0x93E)),
    ("chinese",
     chars(0x4F60, 0x597D, 0xFF0C, 0x4E16, 0x754C, 0x3002),
     chars(0x4F60, 0x597D, 0x4E16, 0x754C)),
    ("japanese",
     chars(0x3053, 0x3093, 0x306B, 0x3061, 0x306F, 0x3001, 0x4E16, 0x754C, 0xFF01),
     chars(0x3053, 0x3093, 0x306B, 0x3061, 0x306F, 0x4E16, 0x754C)),
    ("thai",
     chars(0xE2A, 0xE27, 0xE31, 0xE2A, 0xE14, 0xE35, 0xE04, 0xE23, 0xE31, 0xE1A),
     chars(0xE2A, 0xE27, 0xE31, 0xE2A, 0xE14, 0xE35, 0xE04, 0xE23, 0xE31, 0xE1A)),
    ("armenian",
     chars(0x532, 0x561, 0x580, 0x565, 0x582, 0x55D),
     chars(0x562, 0x561, 0x580, 0x565, 0x582)),
    ("apostrophe-right",
     chars("It", 0x2019, "s"),
     "its"),
    ("apostrophe-left",
     chars("It", 0x2018, "s"),
     "its"),
    ("double-quotes",
     chars(0x201C, "quote", 0x201D),
     "quote"),
    ("guillemets",
     chars(0xAB, "Bonjour", 0xBB),
     "bonjour"),
    ("german-quotes",
     chars(0x201E, "ja", 0x201C),
     "ja"),
    ("modifier-apostrophe",
     chars("don", 0x2BC, "t"),
     chars("don", 0x2BC, "t")),
    ("acute-as-apostrophe",
     chars("don", 0xB4, "t"),
     chars("don", 0xB4, "t")),
    ("okina-letter",
     chars("Hawai", 0x2BB, "i"),
     chars("hawai", 0x2BB, "i")),
    ("okina-quote",
     chars("Hawai", 0x2018, "i"),
     "hawaii"),
    ("primes",
     chars("5", 0x2032, " 10", 0x2033),
     "5 10"),
    ("fullwidth-apostrophe",
     chars("It", 0xFF07, "s"),
     "its"),
    ("hyphen-u2010",
     chars("well", 0x2010, "known"),
     "wellknown"),
    ("nb-hyphen",
     chars("well", 0x2011, "known"),
     "wellknown"),
    ("en-dash",
     chars("1990", 0x2013, "2000"),
     "19902000"),
    ("em-dash-joined",
     chars("done", 0x2014, "gone"),
     "donegone"),
    ("em-dash-spaced",
     chars("done ", 0x2014, " gone"),
     "done gone"),
    ("horizontal-bar",
     chars("a", 0x2015, "b"),
     "ab"),
    ("minus-sign",
     chars(0x2212, "5"),
     chars(0x2212, "5")),
    ("soft-hyphen",
     chars("co", 0xAD, "operate"),
     chars("co", 0xAD, "operate")),
    ("small-hyphen",
     chars("x", 0xFE63, "y"),
     "xy"),
    ("fullwidth-hyphen",
     chars("x", 0xFF0D, "y"),
     "xy"),
    ("wave-dash",
     chars("x", 0x301C, "y"),
     "xy"),
    ("nbsp",
     chars("a", 0xA0, "b"),
     "a b"),
    ("figure-space",
     chars("a", 0x2007, "b"),
     "a b"),
    ("narrow-nbsp",
     chars("a", 0x202F, "b"),
     "a b"),
    ("thin-space",
     chars("a", 0x2009, "b"),
     "a b"),
    ("ideographic-space",
     chars("a", 0x3000, "b"),
     "a b"),
    ("ogham-space",
     chars("a", 0x1680, "b"),
     "a b"),
    ("line-separator",
     chars("a", 0x2028, "b"),
     "a b"),
    ("mongolian-vowel-separator",
     chars("a", 0x180E, "b"),
     chars("a", 0x180E, "b")),
    ("zwsp",
     chars("a", 0x200B, "b"),
     chars("a", 0x200B, "b")),
    ("zwnj",
     chars("a", 0x200C, "b"),
     chars("a", 0x200C, "b")),
    ("zwj",
     chars("a", 0x200D, "b"),
     chars("a", 0x200D, "b")),
    ("word-joiner",
     chars("a", 0x2060, "b"),
     chars("a", 0x2060, "b")),
    ("bom-start",
     chars(0xFEFF, "hello"),
     chars(0xFEFF, "hello")),
    ("arabic-indic-digits",
     chars(0x663, 0x664, 0x665),
     chars(0x663, 0x664, 0x665)),
    ("devanagari-digits",
     chars(0x969, 0x96A, 0x96B),
     chars(0x969, 0x96A, 0x96B)),
    ("fullwidth-digits",
     chars(0xFF13, 0xFF14, 0xFF15),
     chars(0xFF13, 0xFF14, 0xFF15)),
    ("superscript",
     chars("x", 0xB2),
     chars("x", 0xB2)),
    ("circled",
     chars(0x2460, 0x2461),
     chars(0x2460, 0x2461)),
    ("vulgar-half",
     chars(0xBD),
     chars(0xBD)),
    ("roman-numeral",
     chars(0x216B),
     chars(0x217B)),
    ("arabic-decimal",
     chars(0x663, 0x66B, 0x661, 0x664),
     chars(0x663, 0x661, 0x664)),
    ("arabic-thousands",
     chars("1", 0x66C, "000"),
     "1000"),
    ("euro",
     chars(0x20AC, "5"),
     chars(0x20AC, "5")),
    ("pound",
     chars(0xA3, "5"),
     chars(0xA3, "5")),
    ("copyright",
     chars(0xA9, " 2026"),
     chars(0xA9, " 2026")),
    ("trademark",
     chars("werpy", 0x2122),
     chars("werpy", 0x2122)),
    ("degree",
     chars("25", 0xB0, "C"),
     chars("25", 0xB0, "c")),
    ("multiply",
     chars("2", 0xD7, "3"),
     chars("2", 0xD7, "3")),
    ("arrow",
     chars("a", 0x2192, "b"),
     chars("a", 0x2192, "b")),
    ("heart-symbol",
     chars("I ", 0x2665, " NY"),
     chars("i ", 0x2665, " ny")),
    ("emoji",
     chars(0x1F600),
     chars(0x1F600)),
    ("emoji-skin-tone",
     chars(0x1F44D, 0x1F3FD),
     chars(0x1F44D, 0x1F3FD)),
    ("emoji-vs16",
     chars(0x2764, 0xFE0F),
     chars(0x2764, 0xFE0F)),
    ("emoji-zwj-family",
     chars("family ", 0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467, " photo"),
     chars("family ", 0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467, " photo")),
    ("emoji-flag",
     chars(0x1F1E6, 0x1F1FA),
     chars(0x1F1E6, 0x1F1FA)),
    ("keycap-hash",
     chars("#", 0xFE0F, 0x20E3),
     chars(0xFE0F, 0x20E3)),
    ("keycap-one",
     chars("1", 0xFE0F, 0x20E3),
     chars("1", 0xFE0F, 0x20E3)),
    ("section-sign",
     chars(0xA7, " 5"),
     "5"),
    ("bullet",
     chars(0x2022, " item"),
     "item"),
    ("ellipsis",
     chars("wait", 0x2026),
     "wait"),
    ("not-equal-composed",
     chars("a ", 0x2260, " b"),
     chars("a ", 0x2260, " b")),
    ("not-equal-decomposed",
     chars("a =", 0x338, " b"),
     chars("a ", 0x2260, " b")),
    ("fullwidth-latin",
     chars(0xFF21, 0xFF22, 0xFF23),
     chars(0xFF41, 0xFF42, 0xFF43)),
    ("fullwidth-punctuation",
     chars(0xFF01, 0xFF1F),
     ""),
    ("ligature-fi",
     chars(0xFB01, "ne"),
     chars(0xFB01, "ne")),
    ("halfwidth-katakana",
     chars(0xFF76, 0xFF80, 0xFF76, 0xFF85),
     chars(0xFF76, 0xFF80, 0xFF76, 0xFF85)),
    ("kelvin-sign",
     chars(0x212A),
     "k"),
    ("ohm-sign",
     chars(0x2126),
     chars(0x3C9)),
    ("long-s",
     chars(0x17F),
     chars(0x17F)),
    ("sharp-s",
     chars("STRASSE stra", 0xDF, "e"),
     chars("strasse stra", 0xDF, "e")),
    ("capital-sharp-s",
     chars(0x1E9E),
     chars(0xDF)),
    ("titlecase-dz",
     chars(0x1C5),
     chars(0x1C6)),
    ("sigma-words",
     chars(0x3A3, 0x391, 0x3A3, " ", 0x3A3, 0x391, 0x3A3, "."),
     chars(0x3C3, 0x3B1, 0x3C2, " ", 0x3C3, 0x3B1, 0x3C2)),
    ("cherokee",
     chars(0x13A0, 0x13A1),
     chars(0xAB70, 0xAB71)),
    ("surrogate-lone",
     chars("a", 0xD800, "b"),
     chars("a", 0xD800, "b")),
    ("surrogate-low",
     chars(0xDC80),
     chars(0xDC80)),
    ("surrogate-pair-as-two",
     chars(0xD83D, 0xDE00),
     chars(0xD83D, 0xDE00)),
    ("empty",
     "",
     ""),
    ("spaces",
     "   ",
     ""),
    ("whitespace-mixed",
     chars(0x9, 0xA, " ", 0x3000),
     ""),
    ("ascii-punctuation-only",
     "?!...",
     ""),
    ("unicode-punctuation-only",
     chars(0x2026, " ", 0x201C, 0x201D, " ", 0x2014),
     ""),
)


class TestNormalizeRules(unittest.TestCase):
    """
    TestNormalizeRules - Unit tests for the rule of normalize and for the ways through werpy._normalize_kernel.
    """

    def assert_equal_texts(self, texts, results, expected):
        """Assert that each result equals its expected text, and name the first texts that differ."""
        self.assertEqual(len(results), len(texts))
        differ = [(ascii(t)[:60], ascii(r)[:60], ascii(e)[:60]) for t, r, e in zip(texts, results, expected) if r != e]
        self.assertFalse(differ, f"{len(differ)} of {len(texts)} texts differ, the first ones: {differ[:5]}")

    def test_rule_matches_reference(self):
        """
        Test that normalize gives the result of the reference n1 for every code point of the general categories P, S,
        M, Z, Cc, Cf, Cs, Lu and Lt and every code point with a canonical decomposition, each alone and between two
        letters, and for 5,000 seeded random texts of these characters and the ASCII characters, lone surrogates
        included. The texts are normalized as one list, and every 25th text also on its own.
        """
        texts = property_texts()
        expected = [reference.n1(text) for text in texts]
        self.assert_equal_texts(texts, werpy.normalize(texts), expected)
        self.assert_equal_texts(texts[::25], [werpy.normalize(text) for text in texts[::25]], expected[::25])

    def test_idempotent(self):
        """
        Test that a result that is normalized again does not change, for the texts of test_rule_matches_reference.
        """
        once = werpy.normalize(property_texts())
        self.assert_equal_texts(once, werpy.normalize(once), once)

    def test_ascii_text_keeps_its_result(self):
        """
        Test that a text of ASCII characters gives the result of werpy 3.5.0 (the reference today_keep): each of the
        128 ASCII characters alone, between two letters and between two words, and 2,000 seeded random ASCII texts.
        """
        rng = random.Random(SEED)
        characters = [chr(cp) for cp in range(128)]
        texts = characters + ["a" + c + "b" for c in characters] + ["Word " + c + " Word" for c in characters]
        texts += ["".join(rng.choice(characters) for _ in range(rng.randint(1, 40))) for _ in range(2000)]
        self.assert_equal_texts(texts, werpy.normalize(texts), [reference.today_keep(text) for text in texts])
        self.assert_equal_texts(texts, werpy.normalize(texts), [reference.n1(text) for text in texts])

    def test_documented_classes(self):
        """
        Test one text for each row of the table of character classes in the docstring of normalize and on the
        normalization page.
        """
        for row, text, result in DOCUMENTED_CLASSES:
            with self.subTest(row=row):
                self.assertEqual(werpy.normalize(text), result)
                self.assertEqual(reference.n1(text), result)

    def test_canonical_equivalents_equal(self):
        """
        Test that canonically equivalent texts give one result, the result of the reference: e with an acute accent
        as one character and as two; A with a ring above in three forms; not equal to, not less than and not greater
        than, composed and with a combining long solidus overlay; the Greek varia and the grave accent; Hangul
        syllables and their letters; J with a caron; the Kelvin and Ohm signs and their letters; two marks in either
        order.
        """
        sets = {
            "e acute": [chars("caf", 0xE9), chars("cafe", ACUTE)],
            "A ring": [chars(0xC5, "ngstr"), chars(0x212B, "ngstr"), chars("A", 0x30A, "ngstr")],
            "not equal": [chars("a ", 0x2260, " b"), chars("a =", 0x338, " b")],
            "not less, not greater": [chars("a ", 0x226E, 0x226F), chars("a <", 0x338, ">", 0x338)],
            "Greek varia": [chars("x", 0x1FEF, "y"), chars("x", 0x60, "y")],
            "Hangul": [chars(0xD55C, 0xAD6D), chars(0x1112, 0x1161, 0x11AB, 0x1100, 0x116E, 0x11A8)],
            "J caron": [chars("J", CARON), chars(0x1F0)],
            "Kelvin": [chars(0x212A, "elvin"), "Kelvin"],
            "Ohm": [chars(0x2126), chars(0x3A9)],
            "marks out of order": [chars("a", ACUTE, DOT_BELOW), chars("a", DOT_BELOW, ACUTE)],
        }
        for name, texts in sets.items():
            with self.subTest(name=name):
                results = werpy.normalize(texts)
                self.assertEqual(len(set(results)), 1)
                self.assertEqual(results[0], reference.n1(texts[0]))

    def test_lone_surrogates_kept(self):
        """
        Test that lone surrogates are kept, as a str and in a list, a tuple and numpy arrays of dtype U and object:
        one between two letters, a low surrogate alone, two adjacent ones that would pair in UTF-16, two separated by
        a space, one before a capital sigma and a comma, and one before a combining mark.
        """
        texts = [chars("a", 0xD800, "b"), chars(0xDC80), chars(0xD83D, 0xDE00), chars(0xDBFF, " ", 0xDFFF),
                 chars("A", 0xD800, 0x3A3, ","), chars(0xD800, ACUTE)]
        expected = [reference.n1(text) for text in texts]
        self.assertEqual(werpy.normalize(texts[0]), texts[0])
        self.assertEqual(werpy.normalize(texts[2]), texts[2])
        for form in (texts, tuple(texts), *numpy_forms(texts)):
            with self.subTest(form=type(form).__name__):
                self.assertEqual(werpy.normalize(form), expected)
        self.assertEqual([werpy.normalize(text) for text in texts], expected)

    def test_batch_forms(self):
        """
        Test that a list, a tuple and numpy arrays of dtype U and object give the same list of plain str, equal to the
        results of the single texts, and that a str subclass and numpy.str_ give a plain str. Empty batches give an
        empty list.
        """
        texts = ["", " It's A test. ", chars("cafe", ACUTE, " ", 0x2014), chars(SIGMA_WORD, ", x"),
                 chars("a =", 0x338, " b"), chars("a", 0xD800), chars(0xB95, 0xBBE), "\t\n"]
        expected = [reference.n1(text) for text in texts]
        for form in (texts, tuple(texts), *numpy_forms(texts)):
            with self.subTest(form=type(form).__name__, dtype=str(getattr(form, "dtype", ""))):
                result = werpy.normalize(form)
                self.assertIs(type(result), list)
                self.assertEqual(result, expected)
                self.assertEqual({type(text) for text in result}, {str})
        singles = [werpy.normalize(text) for text in texts]
        self.assertEqual(singles, expected)
        numpy_texts = [] if np is None else [np.str_(chars("Caf", 0xC9, ","))]
        for text in (Text(chars("Caf", 0xC9, ",")), *numpy_texts, Text("A, b")):
            with self.subTest(text_type=type(text).__name__):
                self.assertIs(type(werpy.normalize(text)), str)
                self.assertEqual(werpy.normalize(text), reference.n1(str(text)))
        for empty in ([], (), *numpy_forms([])):
            with self.subTest(empty=type(empty).__name__):
                self.assertEqual(werpy.normalize(empty), [])

    def test_element_error(self):
        """
        Test that an element that is not a str raises the WerpyTypeError with its type and index, also after texts that
        are not ASCII and in a numpy array of dtype object.
        """
        cases = [
            (["a", 1], 1),
            ([chars("caf", 0xE9), SIGMA_WORD, chars("a =", 0x338), None], 3),
            ((chars(0xB95, 0xBBE), b"x"), 1),
            *([] if np is None else [(np.array([chars("x", 0x2014), 2.5], dtype=object), 1)]),
        ]
        for batch, index in cases:
            with self.subTest(batch=ascii(list(batch))):
                with self.assertRaises(WerpyTypeError) as context:
                    werpy.normalize(batch)
                self.assertEqual(
                    str(context.exception),
                    f"Every element of text must be a str. An element of type {type(batch[index]).__name__} was found "
                    f"at index {index}.",
                )

    def test_supplementary_punctuation(self):
        """
        Test that punctuation above U+FFFF is removed, alone, in words and next to emoji, which are kept: the Adlam
        initial exclamation and question marks and the Aegean word separator line and dot.
        """
        cases = [
            (chars(0x1E95E), ""),
            (chars(0x10100), ""),
            (chars("a", 0x1E95E, "b"), "ab"),
            (chars("a", 0x10100, "b"), "ab"),
            (chars(EMOJI, 0x1E95E, EMOJI), EMOJI + EMOJI),
            (chars(0x10100, " ", EMOJI, " ", 0x1E95E), EMOJI),
            (chars("x", 0x1E95F, EMOJI, 0x10101, "y"), "x" + EMOJI + "y"),
        ]
        for text, result in cases:
            with self.subTest(text=ascii(text)):
                self.assertEqual(werpy.normalize(text), result)
                self.assertEqual(reference.n1(text), result)

    def test_final_sigma(self):
        """
        Test that a removed punctuation character ends a word for the Greek capital sigma, as an ASCII punctuation
        character does: two words of capital sigmas joined by a comma, an em dash, a right single quotation mark, an
        Adlam initial exclamation mark and an apostrophe each give two words with the final sigma.
        """
        for mark in (",", chr(0x2014), chr(0x2019), chr(0x1E95E), "'"):
            with self.subTest(mark=ascii(mark)):
                self.assertEqual(werpy.normalize(SIGMA_WORD + mark + SIGMA_WORD), FINAL_SIGMA_WORD * 2)
        self.assertEqual(
            werpy.normalize(SIGMA_WORD + " " + SIGMA_WORD + "."), FINAL_SIGMA_WORD + " " + FINAL_SIGMA_WORD
        )

    def test_wer_after_normalize(self):
        """
        Test the Word Error Rate of two texts that differ only in their punctuation and in the form of an accented
        letter: it is 0.0 after normalize. A reference made only of punctuation becomes blank, and wer raises the
        WerpyValueError for a blank reference.
        """
        reference_text = chars("It", 0x2019, "s ", 0x201C, "done", 0x201D, " ", 0x2014, " cafe", ACUTE)
        hypothesis_text = chars("It's \"done\" - caf", 0xE9)
        self.assertEqual(werpy.wer(werpy.normalize(reference_text), werpy.normalize(hypothesis_text)), 0.0)
        self.assertEqual(werpy.wers([reference_text], [hypothesis_text]), [1.0])
        with self.assertRaises(WerpyValueError):
            werpy.wer(werpy.normalize(chars(0x2026, " ", 0x201C, 0x201D, " ", 0x2014)), "x")

    def test_examples_pinned(self):
        """
        Test the results of the example texts of the normalisation design: ASCII punctuation, control characters,
        accents in composed and decomposed form, other scripts, apostrophes, quotation marks, dashes, spaces, format
        characters, digits, symbols and emoji, compatibility characters, case, lone surrogates and blank texts.
        """
        for name, text, result in PINNED_EXAMPLES:
            with self.subTest(name=name):
                self.assertEqual(werpy.normalize(text), result)

    def test_kernel_ways(self):
        """
        Test that each way through werpy._normalize_kernel gives the result of the reference, as a single text and in a
        list, with texts just below, at and above the length of the stack buffers.
        """
        cases = kernel_texts()
        for name, text in cases:
            with self.subTest(name=name):
                self.assertEqual(werpy.normalize(text), reference.n1(text))
        texts = [text for _name, text in cases]
        self.assert_equal_texts(texts, werpy.normalize(texts), [reference.n1(text) for text in texts])

    def test_quick_check_maybe(self):
        """
        Test the quick check of the characters whose NFC_Quick_Check is MAYBE, alone and between two words: the
        composition cases of maybe_texts, and sentences in Tamil, Bengali and Hindi, in NFC and in decomposed form.
        """
        for name, text in maybe_texts():
            for context in (text, "x " + text + " y"):
                with self.subTest(name=name, context=ascii(context)[:40]):
                    self.assertEqual(werpy.normalize(context), reference.n1(context))

    def test_exact_function_agrees(self):
        """
        Test that the passes of the compiled kernel and the step-by-step function of werpy._normalize, which the kernel
        uses only for the texts of the exact set, give the same result on the texts of the property tests and of each
        way through the kernel.
        """
        texts = property_texts() + [text for _name, text in kernel_texts() + maybe_texts()]
        exact = [NORMALIZE._normalize_exact(text) for text in texts]  # pylint: disable=protected-access
        self.assert_equal_texts(texts, werpy.normalize(texts), exact)

    def test_composition_pairs(self):
        """
        Test canonical composition after the second character of every primary composite: each composite as its two
        characters, alone and between two words; each second character after the letter q and after the capital omega;
        5,000 seeded texts of a first character with a mark and a second character, or with two second characters; and
        the Hangul syllables of every leading consonant and vowel, and seeded syllables with a final consonant.
        """
        _singles, composites = code_point_scan()
        rng = random.Random(SEED)
        firsts = sorted({first for _cp, first, _second in composites})
        seconds = sorted({second for _cp, _first, second in composites})
        marks = sorted(set(range(0x300, 0x370)) | {0x5B0, 0x93C, 0x9BC, 0x1DCA, 0x20D2})
        texts = [chr(first) + chr(second) for _cp, first, second in composites]
        texts += ["x " + chr(first) + chr(second) + "y z" for _cp, first, second in composites]
        texts += ["q" + chr(second) for second in seconds] + [chr(0x3A9) + chr(second) for second in seconds]
        for _ in range(2500):
            texts.append(chr(rng.choice(firsts)) + chr(rng.choice(marks)) + chr(rng.choice(seconds)))
            texts.append(chr(rng.choice(firsts)) + chr(rng.choice(seconds)) + chr(rng.choice(seconds)))
        texts += [chr(lead) + chr(vowel) for lead in range(0x1100, 0x1113) for vowel in range(0x1161, 0x1176)]
        texts += [chr(0xAC00 + 28 * rng.randrange(399) + rng.randrange(28)) + chr(rng.randrange(0x11A8, 0x11C3))
                  for _ in range(1000)]
        self.assertGreater(len(composites), 0)
        self.assert_equal_texts(texts, werpy.normalize(texts), [reference.n1(text) for text in texts])

    def test_first_texts_of_a_new_process(self):
        """
        Test that the first texts that a new Python process normalizes give the result of the reference: the first
        text that is not ASCII builds the composition data and the blocks of the table that the texts need.
        """
        texts = [chars(0xB95, 0xBBE, "! ", 0xBC6, 0xBBE), chars(0x1100, 0x1161), SIGMA_WORD + ", " + SIGMA_WORD,
                 chars("a =", 0x338, " b"), chars("CAFE", ACUTE, " ", 0x2014)]
        code = f"import werpy\nprint(ascii([werpy.normalize(text) for text in {ascii(texts)}]))"
        completed = subprocess.run(  # nosec B603
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=300
        )
        self.assertEqual(ast.literal_eval(completed.stdout.strip()), [reference.n1(text) for text in texts])

    def test_exact_and_special_characters(self):
        """
        Test, from the Unicode database of the running Python, that the table of werpy._normalize marks as special
        every character whose lowercase is longer than one character or depends on the characters around it, and
        marks as exact the characters of its exact set, which hold the combining long solidus overlay and the Greek
        varia. Texts with each of these characters give the result of the reference in several contexts.
        """
        # pylint: disable=protected-access
        special = []
        for cp in range(sys.maxunicode + 1):
            c = chr(cp)
            lower = c.lower()
            if len(lower) != 1 or ("A" + c).lower()[1:] != lower or (c + "A").lower()[:-1] != lower:
                special.append(cp)
        _seconds, _firsts, exact = NORMALIZE._composition()
        self.assertIn(0x130, special)
        self.assertIn(0x3A3, special)
        self.assertLessEqual({0x338, 0x1FEF}, set(exact))
        blocks = {}
        for cp in sorted(set(special) | set(exact)):
            data = blocks.setdefault(cp >> 8, NORMALIZE._block(cp >> 8))
            flags = NORMALIZE._ENTRY.unpack_from(data, (cp & 0xFF) * NORMALIZE._ENTRY.size)[1]
            with self.subTest(code_point=f"U+{cp:04X}"):
                self.assertEqual(bool(flags & NORMALIZE._SPECIAL), cp in special)
                self.assertEqual(bool(flags & NORMALIZE._EXACT), cp in exact)
                c = chr(cp)
                texts = [c, "A" + c + "A", "a " + c + ", b", c + c, "x" + c + ACUTE, SIGMA_WORD + c + chr(0x2019)]
                self.assertEqual(werpy.normalize(texts), [reference.n1(text) for text in texts])

    def test_kernel_setup_checks(self):
        """
        Test the checks of the setup functions of werpy._normalize_kernel: an ASCII table that does not hold 128
        entries, composition keys that are not 8-byte integers, and a block of the table that does not hold 256
        entries each raise a ValueError, and normalize gives the same results afterwards.
        """
        # pylint: disable=protected-access
        text = chars("Caf", 0xC9, ", ", SIGMA_WORD)
        before = werpy.normalize(text)
        with self.assertRaises(ValueError) as context:
            KERNEL.setup(NORMALIZE._block, NORMALIZE._normalize_exact, bytes(127))
        self.assertEqual(str(context.exception), "The ASCII table must hold 128 entries.")
        with self.assertRaises(ValueError) as context:
            KERNEL.set_pairs(bytes(7))
        self.assertEqual(str(context.exception), "The composition pairs must be 8-byte keys.")
        KERNEL.setup(lambda _block: b"", NORMALIZE._normalize_exact, NORMALIZE._ascii_table())
        try:
            # The planes 4 to 13 hold no assigned character, so no other test needs their blocks.
            errors = []
            for cp in range(0x40000, 0xE0000, 0x100):
                try:
                    KERNEL.normalize_text(chr(cp))
                except ValueError as error:
                    errors.append(str(error))
                    break
        finally:
            KERNEL.setup(NORMALIZE._block, NORMALIZE._normalize_exact, NORMALIZE._ascii_table())
        self.assertEqual(errors, ["A block of the normalization table must hold 256 entries."])
        self.assertEqual(werpy.normalize(text), before)
        self.assertEqual(werpy.normalize("A, b" + string.whitespace), "a b")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
