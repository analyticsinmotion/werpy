import numpy as np


class Normalize:
    """
    A class that provides the preprocessing methods for normalizing a text input.
    This class transforms text data into the optimal input format for the Word Error Rate (WER) function.

    Attributes
    ----------
    text : str, list or numpy array
        The input text to be normalized. It can be a single string, a list of strings, or a numpy array.

    Methods
    -------
    remove_punctuation
        Removes any non-alphanumeric characters
    convert_to_lowercase
        Converts the given text into lowercase
    replace_multiple_spaces
        Changes any instances of a double space back to a standard single space
    remove_leading_trailing_spaces
        Removes any leading and/or trailing spaces in a text string
    apply_normalization
        Applies all the normalization methods in this class to the input text and outputs an array datatype
    """
    __slots__ = ['text']

    def __init__(self, text) -> None:
        """
        Class Constructor

        Raises
        ------
        TypeError
            if input text is not a str, list or np.ndarray data type.
        """
        try:
            if isinstance(text, (str, list, np.ndarray)):
                self.text = np.array(text)
            else:
                raise TypeError("Input must be a String, List or Numpy Array")
        except TypeError as err:
            print("TypeError:", err)

    def remove_punctuation(self):
        """
        Method that removes punctuation from input text
        """
        punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        self.text = np.char.translate(self.text, str.maketrans('', '', punctuation))

    def convert_to_lowercase(self):
        """
        Method that makes all input text lowercase
        """
        self.text = np.char.lower(self.text)

    def replace_multiple_spaces(self):
        """
        Method that replaces double spaces with a single space in input text
        """
        self.text = np.char.replace(self.text, '  ', ' ')

    def remove_leading_trailing_spaces(self):
        """
        Method that removes leading and trailing spaces
        """
        self.text = np.char.strip(self.text)

    def apply_normalization(self) -> np.ndarray:
        """
        Method that applies all the normalization methods in this class to the input text

        Returns
        -------
        self.text : np.ndarray
            A normalized version of the input text
        """
        self.remove_punctuation()
        self.convert_to_lowercase()
        self.replace_multiple_spaces()
        self.remove_leading_trailing_spaces()
        return self.text


def instantiate_normalize_class(text):
    """
    This function creates an instance of the 'Normalize' object and applies the `apply_normalization` method to its
    text input. As Python Classes cannot directly be vectorized this is a helper function that allows a similar
    capability.

    Parameters
    ----------
    text : str, list or numpy array
        The input text to be normalized. It can be a single string, a list of strings, or a numpy array.

    Returns
    -------
    np.ndarray
        A normalized version of the input text
    """
    obj = Normalize(text)
    return obj.apply_normalization()


def normalize(text):
    """
    The main function for this module. Through the helper function it vectorizes the apply_normalization method and
    produces the final normalized version of the input text.

    Parameters
    ----------
    text : str, list or numpy array
        The input text to be normalized. It can be a single string, a list of strings, or a numpy array.

    Returns
    -------
    list
        Applies vectorization to improve output performance.

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

    Raises
    ------
    TypeError
        if any input data is has an int, float or complex data type.
    AttributeError
        if input text is not a str, list or np.ndarray data type.
    """
    try:
        vectorize_instantiate_normalize_class = np.vectorize(instantiate_normalize_class)
        return vectorize_instantiate_normalize_class(text).tolist()
    except TypeError as err:
        print("TypeError:", err,
              "\nAll text should be in a str(string) format. "
              "Please check your input does not include any Numeric Data Types such as int, float or complex.")
    except AttributeError:
        print(
            "AttributeError: "
            "The normalization method cannot be executed if data input is not a String, List or Numpy Array")
