# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_errorhandler.py

This module contains a set of unit tests for the 'error_handler' function in the 'werpy' package.

The 'error_handler' function validates the reference and hypothesis inputs before any metric is
calculated, and returns them, with a tuple as a list. It raises a WerpyTypeError, which is also an
AttributeError, when an input is not a string, list, tuple or one-dimensional numpy array, when the two
inputs are not of the same kind, or when an element of a list, tuple or numpy array is not a string. It
raises a WerpyValueError, which is also a ValueError and a ZeroDivisionError, when two sequences differ
in length, when two empty sequences are not allowed, or when a reference is blank. The tests in this
module call the function directly so that each exception is observed as raised.

To run the tests, execute this module as the main program.

For more details on the 'error_handler' function and how to use it, please refer to the 'werpy' package
documentation.
"""

import unittest
import numpy as np
from werpy._errorhandler import error_handler


KIND_MESSAGE = "Reference and hypothesis must both be strings, or both be lists, tuples or numpy arrays."
LENGTH_MESSAGE = "The Reference and Hypothesis input parameters must have the same number of elements."
EMPTY_MESSAGE = (
    "Reference and hypothesis must not be empty. The Word Error Rate is not defined for empty sequences."
)
BLANK_STRING_MESSAGE = (
    "Invalid input: reference must not be blank, and reference and hypothesis cannot both be empty."
)


def blank_sequence_message(index):
    """Return the message raised for a blank reference at the given position in a sequence."""
    return f"Invalid input: reference must not be blank. A blank reference was found at index {index}."


def type_message(name, type_name):
    """Return the message raised for a reference or hypothesis of the wrong type."""
    return f"{name} must be a str, or a list, tuple or one-dimensional numpy array of str, not {type_name}."


def element_message(name, type_name, index):
    """Return the message raised for an element of a sequence that is not a string."""
    return f"Every element of {name} must be a str. An element of type {type_name} was found at index {index}."


class TestErrorHandler(unittest.TestCase):
    """
    This class contains unit tests for the 'error_handler' function, which validates the reference and
    hypothesis inputs passed to the werpy metric functions.
    """

    def test_error_handler_valid_strings(self):
        """
        Test that a pair of non-blank strings passes validation and is returned unchanged.
        """
        self.assertEqual(error_handler("i love cold pizza", "i love pizza"), ("i love cold pizza", "i love pizza"))

    def test_error_handler_valid_sequences(self):
        """
        Test that a pair of lists and a pair of numpy arrays of equal length pass validation and are returned
        unchanged, as the same objects.
        """
        for reference, hypothesis in (
            (["a b", "c d"], ["a b", "c e"]),
            (np.array(["a b", "c d"]), np.array(["a b", "c e"])),
        ):
            returned_reference, returned_hypothesis = error_handler(reference, hypothesis)
            self.assertIs(returned_reference, reference)
            self.assertIs(returned_hypothesis, hypothesis)

    def test_error_handler_blank_hypothesis_is_valid(self):
        """
        Test that a blank hypothesis passes validation on both the string and the sequence path, and that the
        inputs are returned.
        """
        self.assertEqual(error_handler("a b", ""), ("a b", ""))
        self.assertEqual(error_handler(["a b"], [""]), (["a b"], [""]))

    def test_error_handler_empty_sequences(self):
        """
        Test that two empty sequences pass validation and are returned, and that they raise a ValueError when
        empty sequences are not allowed.
        """
        self.assertEqual(error_handler([], []), ([], []))
        with self.assertRaises(ValueError) as context:
            error_handler([], [], allow_empty=False)
        self.assertEqual(str(context.exception), EMPTY_MESSAGE)

    def test_error_handler_numeric_reference(self):
        """
        Test that a numeric scalar reference raises an AttributeError.
        """
        with self.assertRaises(AttributeError) as context:
            error_handler(1, "a b")
        self.assertEqual(str(context.exception), type_message("reference", "int"))

    def test_error_handler_numeric_hypothesis(self):
        """
        Test that a numeric scalar hypothesis raises an AttributeError.
        """
        with self.assertRaises(AttributeError) as context:
            error_handler("a b", 2.5)
        self.assertEqual(str(context.exception), type_message("hypothesis", "float"))

    def test_error_handler_none_input(self):
        """
        Test that None in either position raises an AttributeError.
        """
        with self.assertRaises(AttributeError) as context:
            error_handler(None, "a b")
        self.assertEqual(str(context.exception), type_message("reference", "NoneType"))
        with self.assertRaises(AttributeError) as context:
            error_handler("a b", None)
        self.assertEqual(str(context.exception), type_message("hypothesis", "NoneType"))

    def test_error_handler_tuple_input(self):
        """
        Test that tuples are accepted as sequences and are returned as lists.
        """
        reference, hypothesis = error_handler(("a b",), ("a c",))
        self.assertIs(type(reference), list)
        self.assertIs(type(hypothesis), list)
        self.assertEqual((reference, hypothesis), (["a b"], ["a c"]))

    def test_error_handler_string_reference_with_list_hypothesis(self):
        """
        Test that a string reference paired with a list hypothesis raises an AttributeError.
        """
        with self.assertRaises(AttributeError) as context:
            error_handler("a b", ["a b"])
        self.assertEqual(str(context.exception), KIND_MESSAGE)

    def test_error_handler_list_reference_with_string_hypothesis(self):
        """
        Test that a list reference paired with a string hypothesis raises an AttributeError.
        """
        with self.assertRaises(AttributeError) as context:
            error_handler(["a b"], "a b")
        self.assertEqual(str(context.exception), KIND_MESSAGE)

    def test_error_handler_unequal_sequence_lengths(self):
        """
        Test that two sequences of different lengths raise a ValueError.
        """
        with self.assertRaises(ValueError) as context:
            error_handler(["a b", "c d"], ["a b"])
        self.assertEqual(str(context.exception), LENGTH_MESSAGE)

    def test_error_handler_blank_reference_string(self):
        """
        Test that an empty or whitespace-only reference string raises a ZeroDivisionError.
        """
        with self.assertRaises(ZeroDivisionError) as context:
            error_handler("", "a b")
        self.assertEqual(str(context.exception), BLANK_STRING_MESSAGE)
        with self.assertRaises(ZeroDivisionError) as context:
            error_handler("   ", "a b")
        self.assertEqual(str(context.exception), BLANK_STRING_MESSAGE)

    def test_error_handler_both_strings_blank(self):
        """
        Test that a blank reference and a blank hypothesis together raise a ZeroDivisionError.
        """
        with self.assertRaises(ZeroDivisionError) as context:
            error_handler("", "")
        self.assertEqual(str(context.exception), BLANK_STRING_MESSAGE)

    def test_error_handler_blank_reference_in_list(self):
        """
        Test that an empty reference string inside a list raises a ZeroDivisionError naming its index.
        """
        with self.assertRaises(ZeroDivisionError) as context:
            error_handler([""], ["x y"])
        self.assertEqual(str(context.exception), blank_sequence_message(0))

    def test_error_handler_whitespace_reference_in_list(self):
        """
        Test that a whitespace-only reference inside a list raises a ZeroDivisionError naming its index.
        """
        with self.assertRaises(ZeroDivisionError) as context:
            error_handler(["a b", "   "], ["a b", "c d"])
        self.assertEqual(str(context.exception), blank_sequence_message(1))

    def test_error_handler_blank_reference_in_array(self):
        """
        Test that a blank reference inside a numpy array raises a ZeroDivisionError naming its index.
        """
        with self.assertRaises(ZeroDivisionError) as context:
            error_handler(np.array(["a b", "c d", ""]), np.array(["a b", "c d", "e f"]))
        self.assertEqual(str(context.exception), blank_sequence_message(2))

    def test_error_handler_first_blank_reference_index_reported(self):
        """
        Test that the index in the message is that of the first blank reference when several are blank.
        """
        with self.assertRaises(ZeroDivisionError) as context:
            error_handler(["a b", "", ""], ["a b", "c d", "e f"])
        self.assertEqual(str(context.exception), blank_sequence_message(1))

    def test_error_handler_nested_sequence(self):
        """
        Test that a list of lists raises an AttributeError whose message names the first element that is not a
        string.
        """
        with self.assertRaises(AttributeError) as context:
            error_handler([["a b"]], [["a b"]])
        self.assertEqual(str(context.exception), element_message("reference", "list", 0))

    def test_error_handler_sequence_of_integers(self):
        """
        Test that a list of integers raises an AttributeError whose message names the first element that is not
        a string.
        """
        with self.assertRaises(AttributeError) as context:
            error_handler([1, 2], [1, 2])
        self.assertEqual(str(context.exception), element_message("reference", "int", 0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
