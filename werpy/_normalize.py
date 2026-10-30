# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
The _normalize module provides preprocessing methods for normalizing text input to be optimal for the Word Error Rate
(WER) function. The class contains methods for removing punctuation, converting text to lowercase, and removing all
whitespace such as leading/trailing spaces and multiple in-text spaces. 

This module defines the following function:
    - normalize(text)
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING, overload

from ._errorhandler import check_text_input, element_error

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


# Translation table that maps every ASCII punctuation byte to NUL and leaves every other
# byte value unchanged. Built once when the module is imported.
_PUNCTUATION_TRANSLATE_TABLE = bytes(
    0 if c in string.punctuation.encode() else c for c in range(256)
)


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
    This function serves as a versatile text preprocessing tool, designed to transform
    text data into an optimal format for a variety of natural language processing tasks,
    such as calculating the Word Error Rate (WER).

    Its core functionalities encompass removing punctuation, converting text to
    lowercase, and eliminating unnecessary whitespace.

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
    """
    check_text_input("text", text)

    if isinstance(text, str):
        is_string_flag = True
        text = [text]
    else:
        is_string_flag = False

    normalized_text: list[str] = []

    for sentence in text:
        if not isinstance(sentence, str):
            # The texts before this element are normalized, so their number is its index.
            raise element_error("text", sentence, len(normalized_text))
        cleaned_sentence = sentence.encode().translate(_PUNCTUATION_TRANSLATE_TABLE).decode().lower()
        cleaned_sentence = cleaned_sentence.rstrip("\x00").replace("\x00", "")
        cleaned_sentence = " ".join(cleaned_sentence.split())
        normalized_text.append(cleaned_sentence)

    if is_string_flag:
        return normalized_text[0]

    return normalized_text
