# SPDX-FileCopyrightText: 2023-2025 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
This module provides a function for calculating a list of weighted Word Error Rate for each of the reference and
hypothesis texts. It allows varying weights to be assigned to the insertion, deletion and substitution errors.

This module defines the following function:
    - werps(reference, hypothesis)
"""

import numpy as np
from .errorhandler import error_handler


def werps(
    reference,
    hypothesis,
    insertions_weight=1,
    deletions_weight=1,
    substitutions_weight=1,
):
    """
    This function calculates a list of weighted Word Error Rates for each of the reference and hypothesis texts. It
    allows the insertion, deletion and substitution errors to be penalized or weighted at different rates.

    Parameters
    ----------
    reference : str, list or numpy array
        The ground truth transcription of a recorded speech or the expected output of a live speech.
    hypothesis : str, list or numpy array
        The text generated by a speech-to-text algorithm/system which will be compared to the reference text.
    insertions_weight: int or float, optional
        The weight multiplier for an insertion error
    deletions_weight: int or float, optional
        The weight multiplier for a deletion error
    substitutions_weight: int or float, optional
        The weight multiplier for a substitution error

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
        This function will return either a single weighted Word Error Rate (if the input is a pair of strings) or a
        list of weighted Word Error Rates (if the input is a pair of lists) for each of the reference and hypothesis
        texts.

    Examples
    --------
    >>> ref = ['it blocked sight lines of central park', 'her father was an alderman in the city government']
    >>> hyp = ['it blocked sightlines of central park', 'our father was an elder man in the city government']

    >>> werps_example_1 = werps(ref, hyp)
    >>> print(werps_example_1)
    [0.2857142857142857, 0.3333333333333333]

    >>> werps_example_2 = werps(ref, hyp, insertions_weight = 0.5, deletions_weight = 0.5, substitutions_weight = 1)
    >>> print(werps_example_2)
    [0.21428571428571427, 0.2777777777777778]
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
        weighted_insertions = transform_word_error_rate_breakdown[3] * insertions_weight
        weighted_deletions = transform_word_error_rate_breakdown[4] * deletions_weight
        weighted_substitutions = (
            transform_word_error_rate_breakdown[5] * substitutions_weight
        )
        m = transform_word_error_rate_breakdown[2]
    else:
        weighted_insertions = word_error_rate_breakdown[3] * insertions_weight
        weighted_deletions = word_error_rate_breakdown[4] * deletions_weight
        weighted_substitutions = word_error_rate_breakdown[5] * substitutions_weight
        m = word_error_rate_breakdown[2]

    weighted_errors = sum(
        (weighted_insertions, weighted_deletions, weighted_substitutions)
    )
    werps_result = weighted_errors / m

    if isinstance(word_error_rate_breakdown[0], float):
        return werps_result

    return werps_result.tolist()
