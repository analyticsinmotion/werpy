"""
The normalize module provides preprocessing methods for normalizing text input to be optimal for the Word Error Rate
(WER) function. The class contains methods for removing punctuation, converting text to lowercase, and removing all
whitespace such as leading/trailing spaces and multiple in-text spaces. 

This module defines the following function:
    - normalize(text)
"""

import string


def normalize(text):
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
    TypeError
        If the input is not a valid data type such as (int, float, bool, range, dict,
        bytes, bytearray, complex) or if the input contains nested data (e.g., a list of
        lists), the function raises a TypeError.

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
    if isinstance(text, (int, float, bool, range, dict, bytes, bytearray, complex)):
        raise TypeError("Input must be String, List, Tuple, or NumPy Array.")

    if isinstance(text, str):
        is_string_flag = True
        text = [text]
    else:
        is_string_flag = False

    normalized_text = []
    translate_table = [0 if c in string.punctuation.encode() else c for c in range(256)]
    translate_bytes = bytes(translate_table)

    for sentence in text:
        if not isinstance(sentence, str):
            raise TypeError("Input must be String, List, Tuple, or NumPy Array. "
                            "All data types should be flat, have a depth of 1 and "
                            "contain no nested elements.")        
        cleaned_sentence = sentence.encode().translate(translate_bytes).decode().lower()
        cleaned_sentence = cleaned_sentence.rstrip('\x00').replace('\x00', '')
        cleaned_sentence = ' '.join(cleaned_sentence.split())
        normalized_text.append(cleaned_sentence)

    if is_string_flag:
        return normalized_text[0]

    return normalized_text
