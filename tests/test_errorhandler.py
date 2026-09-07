# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_errorhandler.py

This module contains a set of unit tests for the 'error_handler' function in the 'werpy' package.

The 'error_handler' function validates the reference and hypothesis inputs before any metric is
calculated. It raises an AttributeError when an input is not a string, list or numpy array, or when
the two inputs are not of the same kind, a ValueError when two sequences differ in length, and a
ZeroDivisionError when a reference is blank. The tests in this module call the function directly so
that each exception is observed as raised.

To run the tests, execute this module as the main program.

For more details on the 'error_handler' function and how to use it, please refer to the 'werpy' package
documentation.
"""

import unittest
import numpy as np
from werpy.errorhandler import error_handler


TYPE_MESSAGE = (
    "All text should be in a string format. Please check your input does not include any Numeric data types."
)
KIND_MESSAGE = "Reference and hypothesis must both be strings, or both be lists/arrays."
LENGTH_MESSAGE = "The Reference and Hypothesis input parameters must have the same number of elements."
BLANK_STRING_MESSAGE = (
    "Invalid input: reference must not be blank, and reference and hypothesis cannot both be empty."
)


def blank_sequence_message(index):
    """Return the message raised for a blank reference at the given position in a sequence."""
    return f"Invalid input: reference must not be blank. A blank reference was found at index {index}."


class TestErrorHandler(unittest.TestCase):
    """
    This class contains unit tests for the 'error_handler' function, which validates the reference and
    hypothesis inputs passed to the werpy metric functions.
    """

    def test_error_handler_valid_strings(self):
        """
        Test that a pair of non-blank strings passes validation.
        """
        self.assertTrue(error_handler("i love cold pizza", "i love pizza"))

    def test_error_handler_valid_sequences(self):
        """
        Test that a pair of lists and a pair of numpy arrays of equal length pass validation.
        """
        self.assertTrue(error_handler(["a b", "c d"], ["a b", "c e"]))
        self.assertTrue(error_handler(np.array(["a b", "c d"]), np.array(["a b", "c e"])))

    def test_error_handler_blank_hypothesis_is_valid(self):
        """
        Test that a blank hypothesis passes validation on both the string and the sequence path.
        """
        self.assertTrue(error_handler("a b", ""))
        self.assertTrue(error_handler(["a b"], [""]))

    def test_error_handler_empty_sequences(self):
        """
        Test that two empty sequences pass validation.
        """
        self.assertTrue(error_handler([], []))

    def test_error_handler_numeric_reference(self):
        """
        Test that a numeric scalar reference raises an AttributeError.
        """
        with self.assertRaises(AttributeError) as context:
            error_handler(1, "a b")
        self.assertEqual(str(context.exception), TYPE_MESSAGE)

    def test_error_handler_numeric_hypothesis(self):
        """
        Test that a numeric scalar hypothesis raises an AttributeError.
        """
        with self.assertRaises(AttributeError) as context:
            error_handler("a b", 2.5)
        self.assertEqual(str(context.exception), TYPE_MESSAGE)

    def test_error_handler_none_input(self):
        """
        Test that None in either position raises an AttributeError.
        """
        with self.assertRaises(AttributeError) as context:
            error_handler(None, "a b")
        self.assertEqual(str(context.exception), TYPE_MESSAGE)
        with self.assertRaises(AttributeError) as context:
            error_handler("a b", None)
        self.assertEqual(str(context.exception), TYPE_MESSAGE)

    def test_error_handler_tuple_input(self):
        """
        Test that a tuple is not accepted as a sequence and raises an AttributeError.
        """
        with self.assertRaises(AttributeError) as context:
            error_handler(("a b",), ("a b",))
        self.assertEqual(str(context.exception), TYPE_MESSAGE)

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
        Test that a list of lists passes validation. The metric functions raise an AttributeError for this
        input when the elements are split into words.
        """
        self.assertTrue(error_handler([["a b"]], [["a b"]]))

    def test_error_handler_sequence_of_integers(self):
        """
        Test that a list of integers passes validation. The metric functions raise an AttributeError for this
        input when the elements are split into words.
        """
        self.assertTrue(error_handler([1, 2], [1, 2]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
