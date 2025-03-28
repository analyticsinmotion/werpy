# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
This module provides a function for calculating a list of the Word Error Rates for each of the reference and
hypothesis texts.

This module defines the following function:
    - wers(reference, hypothesis)
"""

import numpy as np
from .errorhandler import error_handler


def wers(reference, hypothesis):
    """
    This function calculates a list of the Word Error Rates for each of the reference and hypothesis texts.

    Parameters
    ----------
    reference : str, list or numpy array
        The ground truth transcription of a recorded speech or the expected output of a live speech.
    hypothesis : str, list or numpy array
        The text generated by a speech-to-text algorithm/system which will be compared to the reference text.

    Raises
    ------
    ValueError
        if the two input parameters do not contain the same amount of elements.
    AttributeError
        if input text is not a string, list or np.ndarray data type.
    ZeroDivisionError
        if input in reference is blank or both reference and hypothesis are empty.

    Returns
    -------
    float or list
        This function will return either a single Word Error Rate (if the input is a pair of strings) or a list of Word
        Error Rates (if the input is a pair of lists) for each of the reference and hypothesis texts.

    Example
    --------
    >>> ref = ['no one else could claim that','she cited multiple reasons why']
    >>> hyp = ['no one else could claim that','she sighted multiple reasons why']
    >>> wers_example_1 = wers(ref, hyp)
    >>> print(wers_example_1)
    [0.0, 0.2]
    """
    try:
        word_error_rate_breakdown = error_handler(reference, hypothesis)
    except (ValueError, AttributeError, ZeroDivisionError) as err:
        print(f"{type(err).__name__}: {str(err)}")
        return None
    if isinstance(word_error_rate_breakdown[0], np.ndarray):
        transform_word_error_rate_breakdown = np.transpose(
            word_error_rate_breakdown.tolist()
        )
        wers_result = transform_word_error_rate_breakdown[0].tolist()
    else:
        wers_result = word_error_rate_breakdown[0]
    return wers_result
