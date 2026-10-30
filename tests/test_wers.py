# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_wers.py

This module contains a set of unit tests for the 'wers' function in the 'werpy' package.

The 'wers' function calculates the Word Error Rate (WER) for each text sequence, 
rather than a collective word error rate across all sequences. 
It takes as input a reference text and a hypothesis text, both as sequences of words, 
and returns the WER for that individual sequence pair. This allows for a more granular, 
sequence-by-sequence error analysis, as opposed to an aggregated WER across all samples. 
By providing per-sequence WERs, the 'wers' function enables better understanding of model 
performance on specific examples.

To run the tests, execute this module as the main program.

For more details on the 'wers' function and how to use it, please refer to the 'werpy' package documentation.

Note: If the '_wers' module is not imported successfully, an ImportError is raised 
to ensure that the required module is available for testing.
"""

import contextlib
import io
import unittest
import numpy as np
from werpy._wers import wers
from werpy.exceptions import WerpyError, WerpyTypeError, WerpyValueError


class TestWers(unittest.TestCase):
    """
    This class contains unit tests for the 'wers' function, which calculates the Word Error Rate (WER)
    between reference and hypothesis text sequences.
    """

    def test_wers_example_1(self):
        """
        Test the wers function with a simple example.

        This test checks the WERS function with a basic example of reference and hypothesis strings
        and ensures that the calculated WER, returned in a list, matches the expected result.
        """
        self.assertEqual(wers("i love cold pizza", "i love pizza"), [0.25])

    def test_wers_example_2(self):
        """
        Test the wers function with multiple reference and hypothesis strings.

        This test evaluates the WERS function with multiple reference and hypothesis text sequences.
        It verifies that the calculated WER aligns with the expected results.

        """
        ref = ["i love cold pizza", "the sugar bear character was popular"]
        hyp = ["i love pizza", "the sugar bare character was popular"]
        expected_result = [0.25, 0.16666666666666666]

        self.assertEqual(wers(ref, hyp), expected_result)

    def test_wers_example_3(self):
        """
        Test the wers function with multiple reference and hypothesis strings.

        This test evaluates the WERS function with multiple reference and hypothesis text sequences.
        It verifies that the calculated WER aligns with the expected results.

        """
        ref = ["no one else could claim that", "she cited multiple reasons why"]
        hyp = ["no one else could claim that", "she sighted multiple reasons why"]
        expected_result = [0.0, 0.2]

        self.assertEqual(wers(ref, hyp), expected_result)

    def test_wers_example_4(self):
        """
        Test the wers function with longer text sequences.

        This test assesses the WERS function's performance with longer reference and hypothesis text sequences.
        It verifies that the calculated WER closely matches the expected results.

        """
        ref = [
            "it is consumed domestically and exported to other countries",
            "the sugar bear character was popular enough to have occasional premium toys",
            "it is one of the most watched television networks in the country",
            "it could be carried and prepared by the individual soldier",
            "he was executed in a lubyanka prison cellar",
            "rufino street in makati right inside the makati central business district",
            "its estuary is considered to have abnormally low rates of dissolved oxygen",
            "he later cited his first wife anita as the inspiration for the song",
            "gadya is the nearest rural locality",
            "taxes are a tool in the adjustment of the economy",
        ]
        hyp = [
            "it is consumed domestically and exported to other countries",
            "the sugar bare character was popular enough to have occasional premium toys",
            "it is one of the most watched television networks in the country",
            "it could be carried and prepared by the individual soldier",
            "he was executed in alabianca prison seller",
            "rofino street in mccauti right inside the macasi central business district",
            "its estiary is considered to have a normally low rates of dissolved oxygen",
            "he later sighted his first wife anita as the inspiration for the song",
            "gadia is the nearest rural locality",
            "taxes are a tool in the adjustment of the economy",
        ]
        expected_result = [
            0.0,
            0.08333333333333333,
            0.0,
            0.0,
            0.375,
            0.2727272727272727,
            0.25,
            0.07692307692307693,
            0.16666666666666666,
            0.0,
        ]

        self.assertEqual(wers(ref, hyp), expected_result)

    def test_wers_example_5(self):
        """
        Test the wers function with numerical reference and hypothesis inputs.

        This test evaluates the WERS function with numerical reference and hypothesis inputs.
        It verifies that the numerical elements raise a WerpyTypeError, which is also an AttributeError, with the
        message that names the reference, the type of the first element and its index, and that nothing is printed.

        """
        ref = [1, 2, 3, 4]
        hyp = [2, 3, 3, 3]

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(AttributeError) as context:
                wers(ref, hyp)

        self.assertIs(type(context.exception), WerpyTypeError)
        self.assertEqual(
            str(context.exception),
            "Every element of reference must be a str. An element of type int was found at index 0.",
        )
        self.assertEqual(buffer.getvalue(), "")

    def test_wers_blank_reference_in_list(self):
        """
        Test the wers function with a list containing a blank reference.

        It verifies that the function raises a WerpyValueError whose message names the index of the blank reference,
        and that nothing is printed.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(WerpyValueError) as context:
                wers(["", "a b"], ["x y", "a b"])

        self.assertIs(type(context.exception), WerpyValueError)
        self.assertEqual(
            str(context.exception),
            "Invalid input: reference must not be blank. A blank reference was found at index 0.",
        )
        self.assertEqual(buffer.getvalue(), "")

    def test_wers_empty_sequences(self):
        """
        Test the wers function with two empty lists.

        It verifies that two empty lists yield an empty list.
        """
        self.assertEqual(wers([], []), [])

    def test_wers_tuple_input(self):
        """
        Test the wers function with reference and hypothesis given as tuples.

        It verifies that tuples are accepted like lists: the function returns the result of the same texts given
        as lists.
        """
        result = wers(("i love cold pizza", "i love pizza"), ("i love pizza", "i love pizza"))

        self.assertEqual(result, [0.25, 0.0])
        self.assertEqual(result, wers(["i love cold pizza", "i love pizza"], ["i love pizza", "i love pizza"]))

    def test_wers_string_pair_returns_list(self):
        """
        Test the wers function with a pair of strings.

        It verifies that the function returns a list that holds one float, equal to the result for the same pair
        given as two one-element lists.
        """
        result = wers("i love cold pizza", "i love pizza")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIs(type(result[0]), float)
        self.assertEqual(result, wers(["i love cold pizza"], ["i love pizza"]))

    def test_wers_raises_werpy_exceptions(self):
        """
        Test the wers function with a string reference and a list hypothesis, and with lists of different lengths.

        It verifies that the function raises a WerpyTypeError and a WerpyValueError with the messages of the input
        validation, that each exception is a WerpyError and an instance of the type that werpy 3.x raised for the
        same input, and that nothing is printed.
        """
        cases = [
            (
                "i love pizza",
                ["i love pizza"],
                WerpyTypeError,
                AttributeError,
                "Reference and hypothesis must both be strings, or both be lists, tuples or numpy arrays.",
            ),
            (
                ["i love pizza"],
                ["i love pizza", "a b"],
                WerpyValueError,
                ValueError,
                "The Reference and Hypothesis input parameters must have the same number of elements.",
            ),
        ]
        for ref, hyp, error, legacy_error, message in cases:
            with self.subTest(error=error.__name__):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    with self.assertRaises(error) as context:
                        wers(ref, hyp)

                self.assertIs(type(context.exception), error)
                self.assertIsInstance(context.exception, WerpyError)
                self.assertIsInstance(context.exception, legacy_error)
                self.assertEqual(str(context.exception), message)
                self.assertEqual(buffer.getvalue(), "")

    def test_wers_numpy_string_array(self):
        """
        Test the wers function with reference and hypothesis given as numpy arrays of string dtype.
        """
        ref = np.array(["i love cold pizza", "the sugar bear character was popular"])
        hyp = np.array(["i love pizza", "the sugar bare character was popular"])
        expected_result = [0.25, 0.16666666666666666]

        self.assertEqual(wers(ref, hyp), expected_result)

    def test_wers_numpy_object_array(self):
        """
        Test the wers function with reference and hypothesis given as numpy arrays of object dtype.
        """
        ref = np.array(["i love cold pizza", "the sugar bear character was popular"], dtype=object)
        hyp = np.array(["i love pizza", "the sugar bare character was popular"], dtype=object)
        expected_result = [0.25, 0.16666666666666666]

        self.assertEqual(wers(ref, hyp), expected_result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
