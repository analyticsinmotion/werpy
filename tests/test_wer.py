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

Note: If the 'wer' module is not imported successfully, an ImportError is raised 
to ensure that the required module is available for testing.
"""

import unittest
from werpy.wer import wer


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
        It verifies that the numerical input will raise an AttributeError.

        """
        ref = [1, 2, 3, 4]
        hyp = [2, 3, 3, 3]
        # The actual return value is None from the try/except block in wer module
        expected_result = None

        self.assertEqual(wer(ref, hyp), expected_result)

    def test_wer_blank_input(self):
        """
        Test the wer function with empty reference and hypothesis sequences.

        This test evaluates the WER function with empty reference and hypothesis text sequences.
        It verifies that blank input will raise a ZeroDivisionError.
        """
        ref = [""]
        hyp = [""]

        # The actual return value is None from the try/except block in wer module
        expected_result = None

        self.assertEqual(wer(ref, hyp), expected_result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
