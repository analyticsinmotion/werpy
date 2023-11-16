"""
This module provides a summary function to display a complete breakdown of the calculated results, returned in a 
DataFrame.

This module defines the following function:
    - summary(reference, hypothesis)
"""

import numpy as np
import pandas as pd
from .errorhandler import error_handler


def summary(reference, hypothesis):
    """
    This function provides a comprehensive breakdown of the calculated results including the WER, Levenshtein 
    Distance and all the insertion, deletion and substitution errors.

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

    Returns
    -------
    pandas.core.frame.DataFrame
        Returns a dataframe containing the following nine columns:
            wer - The Word Error Rate
            ld - The Levenshtein distance
            m - The number of words in the reference sequence
            insertions - count of words that are present in the hypothesis sequence but not in the reference
            deletions - count of words that are present in the reference sequence but not in the hypothesis
            substitutions - count of words needing to be transformed so the hypothesis matches the reference
            inserted_words - list of inserted words
            deleted_words - list of deleted words
            substituted_words - list of substitutions. Each substitution will be shown as a tuple with the reference
            word and the hypothesis word. For example: [(cited, sighted), (abnormally, normally)]
    """
    try:
        word_error_rate_breakdown = error_handler(reference, hypothesis)
    except (ValueError, AttributeError) as err:
        print(f"{type(err).__name__}: {str(err)}")
        return None    
    if isinstance(word_error_rate_breakdown[0], np.ndarray):
        word_error_rate_breakdown = word_error_rate_breakdown.tolist()
    else:
        word_error_rate_breakdown = [word_error_rate_breakdown.tolist()]
    columns = ['wer', 'ld', 'm', 'insertions', 'deletions', 'substitutions', 'inserted_words', 'deleted_words',
                   'substituted_words']    
    df = pd.DataFrame(word_error_rate_breakdown, columns=columns)
    return df
