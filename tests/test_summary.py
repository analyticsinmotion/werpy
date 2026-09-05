# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_summary.py

This module contains a set of unit tests for the 'summary' function in the 'werpy' package.

The 'summary' function calculates the Word Error Rate (WER) and alignment on an individual 
sequence basis rather than aggregated across a corpus. It takes as input a reference text 
sequence and hypothesis text sequence as lists of words. The output is a dataframe containing 
the WER, number of insertions, deletions and substitutions, and a listing of the specific word 
errors (insertions, deletions, substitutions) required to align the hypothesis to the reference 
for that sequence pair. This provides transparent per-sequence WER analysis and error inspection 
rather than just an overall corpus-level WER.

To run the tests, execute this module as the main program.

For more details on the 'summary' function and how to use it, please refer to the 'werpy' package documentation.

Note: If the 'summary' module is not imported successfully, an ImportError is raised 
to ensure that the required module is available for testing.
"""

import contextlib
import io
import unittest
import numpy as np
import pandas as pd
from werpy.summary import summary


class TestSummary(unittest.TestCase):
    """
    This class contains unit tests for the 'summary' function, which calculates the Word Error Rate (WER)
    between reference and hypothesis text sequences, displaying the summary results as a dataframe.
    """

    def test_summary_example_1(self):
        """
        Test the summary function with multiple reference and hypothesis strings.

        This test evaluates the SUMMARY function with multiple reference and hypothesis text sequences.
        It verifies that the calculated SUMMARY dataframe aligns with the expected output dataframe.

        """
        ref = [
            "it is consumed domestically and exported to other countries",
            "rufino street in makati right inside the makati central business district",
            "its estuary is considered to have abnormally low rates of dissolved oxygen",
            "he later cited his first wife anita as the inspiration for the song",
            "no one else could claim that",
        ]
        hyp = [
            "it is consumed domestically and exported to other countries",
            "rofino street in mccauti right inside the macasi central business district",
            "its estiary is considered to have a normally low rates of dissolved oxygen",
            "he later sighted his first wife anita as the inspiration for the song",
            "no one else could claim that",
        ]

        # Generate the actual_result DataFrame
        actual_result = summary(ref, hyp)

        expected_result = pd.DataFrame(
            {
                "wer": [0.000000, 0.272727, 0.250000, 0.076923, 0.000000],
                "ld": [0, 3, 3, 1, 0],
                "m": [9, 11, 12, 13, 6],
                "insertions": [0, 0, 1, 0, 0],
                "deletions": [0, 0, 0, 0, 0],
                "substitutions": [0, 3, 2, 1, 0],
                "inserted_words": [[], [], ["a"], [], []],
                "deleted_words": [[], [], [], [], []],
                "substituted_words": [
                    [],
                    [("rufino", "rofino"), ("makati", "mccauti"), ("makati", "macasi")],
                    [("estuary", "estiary"), ("abnormally", "normally")],
                    [("cited", "sighted")],
                    [],
                ],
            }
        )
        # Set the data type of the "ld" column to int64
        # expected_result['ld'] = expected_result['ld'].astype('int32')

        pd.testing.assert_frame_equal(expected_result, actual_result)

    def test_summary_example_2(self):
        """
        Test the summary function with numerical reference and hypothesis inputs.

        This test evaluates the SUMMARY function with numerical reference and hypothesis inputs.
        It verifies that the numerical input will raise an AttributeError.

        """
        ref = [1, 2, 3, 4]
        hyp = [2, 3, 3, 3]
        # The actual return value is None from the try/except block in wer module
        expected_result = None

        self.assertEqual(summary(ref, hyp), expected_result)

    def test_summary_single_sequence(self):
        """
        Test the summary function with a single reference and hypothesis sequence.
        """
        ref = "this is a test"
        hyp = "this is the test"

        actual_result = summary(ref, hyp)

        expected_result = pd.DataFrame(
            {
                "wer": [0.25],
                "ld": [1],
                "m": [4],
                "insertions": [0],
                "deletions": [0],
                "substitutions": [1],
                "inserted_words": [[]],
                "deleted_words": [[]],
                "substituted_words": [[('a', 'the')]],
            }
        )

        pd.testing.assert_frame_equal(expected_result, actual_result)

    def test_summary_insertion_only(self):
        """
        Test the summary function with a hypothesis that adds one word to the reference.

        It verifies that the added word is counted and listed as an insertion and nothing else.
        """
        actual_result = summary("the cat sat", "the big cat sat")

        expected_result = pd.DataFrame(
            {
                "wer": [0.3333333333333333],
                "ld": [1],
                "m": [3],
                "insertions": [1],
                "deletions": [0],
                "substitutions": [0],
                "inserted_words": [["big"]],
                "deleted_words": [[]],
                "substituted_words": [[]],
            }
        )

        pd.testing.assert_frame_equal(expected_result, actual_result)

    def test_summary_deletion_only(self):
        """
        Test the summary function with a hypothesis that omits one word of the reference.

        It verifies that the omitted word is counted and listed as a deletion and nothing else.
        """
        actual_result = summary("the big cat sat", "the cat sat")

        expected_result = pd.DataFrame(
            {
                "wer": [0.25],
                "ld": [1],
                "m": [4],
                "insertions": [0],
                "deletions": [1],
                "substitutions": [0],
                "inserted_words": [[]],
                "deleted_words": [["big"]],
                "substituted_words": [[]],
            }
        )

        pd.testing.assert_frame_equal(expected_result, actual_result)

    def test_summary_numpy_string_array(self):
        """
        Test the summary function with reference and hypothesis given as numpy arrays of string dtype.
        """
        ref = np.array(["i love cold pizza", "the sugar bear character was popular"])
        hyp = np.array(["i love pizza", "the sugar bare character was popular"])

        actual_result = summary(ref, hyp)

        expected_result = pd.DataFrame(
            {
                "wer": [0.25, 0.16666666666666666],
                "ld": [1, 1],
                "m": [4, 6],
                "insertions": [0, 0],
                "deletions": [1, 0],
                "substitutions": [0, 1],
                "inserted_words": [[], []],
                "deleted_words": [["cold"], []],
                "substituted_words": [[], [("bear", "bare")]],
            }
        )

        pd.testing.assert_frame_equal(expected_result, actual_result)

    def test_summary_numpy_object_array(self):
        """
        Test the summary function with reference and hypothesis given as numpy arrays of object dtype.
        """
        ref = np.array(["i love cold pizza", "the sugar bear character was popular"], dtype=object)
        hyp = np.array(["i love pizza", "the sugar bare character was popular"], dtype=object)

        actual_result = summary(ref, hyp)

        expected_result = pd.DataFrame(
            {
                "wer": [0.25, 0.16666666666666666],
                "ld": [1, 1],
                "m": [4, 6],
                "insertions": [0, 0],
                "deletions": [1, 0],
                "substitutions": [0, 1],
                "inserted_words": [[], []],
                "deleted_words": [["cold"], []],
                "substituted_words": [[], [("bear", "bare")]],
            }
        )

        pd.testing.assert_frame_equal(expected_result, actual_result)

    def test_summary_blank_reference_in_list(self):
        """
        Test the summary function with a list containing a blank reference.

        It verifies that the function prints a message naming the index of the blank reference and returns None.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = summary(["this is a test", ""], ["this is the test", "a b"])

        self.assertIsNone(result)
        self.assertEqual(
            buffer.getvalue().strip(),
            "ZeroDivisionError: Invalid input: reference must not be blank. A blank reference was found at index 1.",
        )

    def test_summary_empty_sequences(self):
        """
        Test the summary function with two empty lists.

        It verifies that two empty lists yield a dataframe with the nine result columns and no rows.
        """
        actual_result = summary([], [])

        expected_columns = [
            "wer",
            "ld",
            "m",
            "insertions",
            "deletions",
            "substitutions",
            "inserted_words",
            "deleted_words",
            "substituted_words",
        ]

        self.assertEqual(list(actual_result.columns), expected_columns)
        self.assertEqual(len(actual_result), 0)

    def test_summary_nested_list_input(self):
        """
        Test the summary function with reference and hypothesis given as lists of lists.

        It verifies that the function prints a message and returns None.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = summary([["this is a test"]], [["this is the test"]])

        self.assertIsNone(result)
        self.assertEqual(buffer.getvalue().strip(), "AttributeError: 'list' object has no attribute 'split'")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
