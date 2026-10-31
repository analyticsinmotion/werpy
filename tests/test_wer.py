# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_wer.py

This module contains a set of unit tests for the 'wer' function in the 'werpy' package. 
The 'wer' function is responsible for calculating the Word Error Rate (WER) between 
reference and hypothesis text sequences.

The module defines the 'TestWer' class, which includes multiple test methods to ensure 
the correctness and reliability of the 'wer' function. These tests cover various scenarios, 
including simple comparisons, multiple reference and hypothesis strings, and longer text sequences.
Each test case compares the output of the 'wer' function with the expected word error rate 
and raises an AssertionError if the actual output does not match the expected values.

The 'wer' function is a crucial component for evaluating the accuracy of text recognition and 
transcription systems. These unit tests aim to validate the accuracy of the WER calculations and 
provide confidence in the function's performance.

To run the tests, execute this module as the main program.

For more details on the 'wer' function and how to use it, please refer to the 'werpy' package documentation.

Note: If the '_wer' module is not imported successfully, an ImportError is raised 
to ensure that the required module is available for testing.
"""

import contextlib
import io
import unittest
import numpy as np
import werpy
from werpy._metrics import calculations_wer_only  # pylint: disable=no-name-in-module
from werpy._wer import wer
from werpy.exceptions import WerpyError, WerpyTypeError, WerpyValueError


class TestWer(unittest.TestCase):
    """
    This class contains unit tests for the 'wer' function, which calculates the Word Error Rate (WER)
    between reference and hypothesis text sequences.
    """

    def test_wer_example_1(self):
        """
        Test the wer function with a simple example.

        This test checks the WER function with a basic example of reference and hypothesis strings
        and ensures that the calculated WER matches the expected result.
        """
        self.assertEqual(wer("i love cold pizza", "i love pizza"), 0.25)

    def test_wer_example_2(self):
        """
        Test the wer function with multiple reference and hypothesis strings.

        This test evaluates the WER function with multiple reference and hypothesis text sequences.
        It verifies that the calculated WER aligns with the expected result.

        """
        ref = ["i love cold pizza", "the sugar bear character was popular"]
        hyp = ["i love pizza", "the sugar bare character was popular"]
        expected_result = 0.2

        self.assertEqual(wer(ref, hyp), expected_result)

    def test_wer_example_3(self):
        """
        Test the wer function with longer text sequences.

        This test assesses the WER function's performance with longer reference and hypothesis text sequences.
        It verifies that the calculated WER closely matches the expected result.

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
        expected_result = 0.11650485436893204

        self.assertEqual(wer(ref, hyp), expected_result)

    def test_wer_example_4(self):
        """
        Test the wer function with numerical reference and hypothesis inputs.

        This test evaluates the WER function with numerical reference and hypothesis inputs.
        It verifies that the numerical elements raise a WerpyTypeError, which is also an AttributeError, with the
        message that names the reference, the type of the first element and its index, and that nothing is printed.

        """
        ref = [1, 2, 3, 4]
        hyp = [2, 3, 3, 3]

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(AttributeError) as context:
                wer(ref, hyp)

        self.assertIs(type(context.exception), WerpyTypeError)
        self.assertEqual(
            str(context.exception),
            "Every element of reference must be a str. An element of type int was found at index 0.",
        )
        self.assertEqual(buffer.getvalue(), "")

    def test_wer_blank_input(self):
        """
        Test the wer function with empty reference and hypothesis sequences.

        This test evaluates the WER function with empty reference and hypothesis text sequences.
        It verifies that the blank reference raises a WerpyValueError that names its index, and that nothing is
        printed.
        """
        ref = [""]
        hyp = [""]

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(WerpyValueError) as context:
                wer(ref, hyp)

        self.assertIs(type(context.exception), WerpyValueError)
        self.assertEqual(
            str(context.exception),
            "Invalid input: reference must not be blank. A blank reference was found at index 0.",
        )
        self.assertEqual(buffer.getvalue(), "")

    def test_wer_blank_reference_in_list(self):
        """
        Test the wer function with a list containing a blank reference.

        It verifies that the function raises a WerpyValueError whose message names the index of the blank reference,
        and that nothing is printed.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(WerpyValueError) as context:
                wer([""], ["x y"])

        self.assertIs(type(context.exception), WerpyValueError)
        self.assertEqual(
            str(context.exception),
            "Invalid input: reference must not be blank. A blank reference was found at index 0.",
        )
        self.assertEqual(buffer.getvalue(), "")

    def test_wer_blank_hypothesis(self):
        """
        Test the wer function with a blank hypothesis on both the string and the list path.

        It verifies that a blank hypothesis is valid input and yields a Word Error Rate of 1.0.
        """
        self.assertEqual(wer("i love pizza", ""), 1.0)
        self.assertEqual(wer(["i love pizza"], [""]), 1.0)

    def test_wer_empty_sequences(self):
        """
        Test the wer function with two empty lists.

        It verifies that two empty lists raise a WerpyValueError with the message that the Word Error Rate is not
        defined for empty sequences, and that nothing is printed.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(WerpyValueError) as context:
                wer([], [])

        self.assertIs(type(context.exception), WerpyValueError)
        self.assertEqual(
            str(context.exception),
            "Reference and hypothesis must not be empty. The Word Error Rate is not defined for empty sequences.",
        )
        self.assertEqual(buffer.getvalue(), "")

    def test_wer_none_input(self):
        """
        Test the wer function with None as the reference.

        It verifies that the function raises a WerpyTypeError whose message names the reference and its type, and
        that nothing is printed.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(WerpyTypeError) as context:
                wer(None, "i love pizza")

        self.assertIs(type(context.exception), WerpyTypeError)
        self.assertEqual(
            str(context.exception),
            "reference must be a str, or a list, tuple or one-dimensional numpy array of str, not NoneType.",
        )
        self.assertEqual(buffer.getvalue(), "")

    def test_wer_raises_werpy_exceptions(self):
        """
        Test the wer function with a string reference and a list hypothesis, and with lists of different lengths.

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
                        wer(ref, hyp)

                self.assertIs(type(context.exception), error)
                self.assertIsInstance(context.exception, WerpyError)
                self.assertIsInstance(context.exception, legacy_error)
                self.assertEqual(str(context.exception), message)
                self.assertEqual(buffer.getvalue(), "")

    def test_wer_numpy_string_array(self):
        """
        Test the wer function with reference and hypothesis given as numpy arrays of string dtype.
        """
        ref = np.array(["i love cold pizza", "the sugar bear character was popular"])
        hyp = np.array(["i love pizza", "the sugar bare character was popular"])
        expected_result = 0.2

        self.assertEqual(wer(ref, hyp), expected_result)

    def test_wer_numpy_object_array(self):
        """
        Test the wer function with reference and hypothesis given as numpy arrays of object dtype.
        """
        ref = np.array(["i love cold pizza", "the sugar bear character was popular"], dtype=object)
        hyp = np.array(["i love pizza", "the sugar bare character was popular"], dtype=object)
        expected_result = 0.2

        self.assertEqual(wer(ref, hyp), expected_result)

    def test_wer_identical_pair(self):
        """
        Test the wer and wers functions and the compiled calculations_wer_only function with an identical pair of
        strings.

        An identical pair has no edits. It verifies that wer returns 0.0, that wers returns [0.0], and that
        calculations_wer_only returns a Word Error Rate of 0.0, an edit distance of 0.0 and the number of words in
        the reference, 4.0.
        """
        reference = "i love cold pizza"
        hypothesis = "i love cold pizza"

        self.assertEqual(wer(reference, hypothesis), 0.0)
        self.assertEqual(werpy.wers(reference, hypothesis), [0.0])
        self.assertEqual(calculations_wer_only(reference, hypothesis).tolist(), [0.0, 0.0, 4.0])

    def test_werpy_public_surface(self):
        """
        Test that the names exported by the werpy package are exactly the documented public functions, that
        each function can be imported from the package by name, and that error_handler and metrics are not
        attributes of the package.
        """
        # pylint: disable=import-outside-toplevel,redefined-outer-name,reimported
        from werpy import normalize, summary, summaryp, wer, werp, werps, wers

        expected_names = [
            "normalize",
            "summary",
            "summaryp",
            "wer",
            "werp",
            "werps",
            "wers",
        ]
        imported = [normalize, summary, summaryp, wer, werp, werps, wers]

        self.assertEqual(sorted(werpy.__all__), expected_names)
        for name, function in zip(expected_names, imported):
            self.assertTrue(callable(getattr(werpy, name)))
            self.assertIs(function, getattr(werpy, name))
        for name in ("error_handler", "metrics"):
            self.assertFalse(hasattr(werpy, name), msg=name)

    def test_werpy_implementation_modules_are_private(self):
        """
        Test that the modules that implement the public functions are private, so that an import of a module
        path named after a function raises ModuleNotFoundError.
        """
        # pylint: disable=import-outside-toplevel,redefined-outer-name,reimported,unused-import
        with self.assertRaises(ModuleNotFoundError):
            import werpy.wer
        with self.assertRaises(ModuleNotFoundError):
            from werpy.wer import wer


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
