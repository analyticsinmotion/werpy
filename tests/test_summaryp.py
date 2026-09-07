# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_summaryp.py

This module contains a set of unit tests for the 'summaryp' function in the 'werpy' package.

The 'summaryp' function offers the capability to incorporate penalty weights for individual 
errors and displays the weighted Word Error Rate (WER) at the conclusion of its analysis. 
This function computes the Word Error Rate (WER) and alignment on a per-sequence basis, 
rather than aggregating across a corpus. It takes reference and hypothesis text sequences 
as input, represented as lists of words. The output is a dataframe that includes WER, the 
count of insertions, deletions, and substitutions, along with a detailed breakdown of specific 
word errors (insertions, deletions, substitutions) necessary to align the hypothesis with the 
reference for each sequence pair. This enhancement ensures transparent per-sequence WER analysis 
and facilitates error inspection, going beyond the conventional corpus-level WER assessment.

To run the tests, execute this module as the main program.

For more details on the 'summaryp' function and how to use it, please refer to the 'werpy' package documentation.

Note: If the 'summaryp' module is not imported successfully, an ImportError is raised 
to ensure that the required module is available for testing.
"""

import contextlib
import io
import unittest
import pandas as pd
from werpy.summaryp import summaryp


class TestSummaryp(unittest.TestCase):
    """
    This class contains unit tests for the 'summaryp' function, which calculates the Word Error Rate (WER)
    between reference and hypothesis text sequences, displaying the summary results as a dataframe.
    """

    def test_summaryp_example_1(self):
        """
        Test the summaryp function with multiple reference and hypothesis strings.

        This test evaluates the SUMMARYP function with multiple reference and hypothesis text sequences.
        It verifies that the calculated SUMMARYP dataframe aligns with the expected output dataframe.

        """
        ref = [
            "the tower caused minor discontent because it blocked sight lines of central park",
            "her father was an alderman in the city government",
            "he was commonly referred to as the blacksmith of ballinalee",
        ]
        hyp = [
            "the tower caused minor discontent because it blocked sightlines of central park",
            "our father was an alderman in the city government",
            "he was commonly referred to as the blacksmith of balen alley",
        ]

        # Generate the actual_result DataFrame
        actual_result = summaryp(
            ref,
            hyp,
            insertions_weight=0.5,
            deletions_weight=0.5,
            substitutions_weight=1,
        )

        data = {
            "wer": [0.15384615384615385, 0.1111111111111111, 0.2],
            "werp": [0.11538461538461539, 0.1111111111111111, 0.15],
            "ld": [2, 1, 2],
            "m": [13, 9, 10],
            "insertions": [0, 0, 1],
            "deletions": [1, 0, 0],
            "substitutions": [1, 1, 1],
            "inserted_words": [[], [], ["balen"]],
            "deleted_words": [["sight"], [], []],
            "substituted_words": [
                [("lines", "sightlines")],
                [("her", "our")],
                [("ballinalee", "alley")],
            ],
        }

        expected_result = pd.DataFrame(data)

        # Set the data type of the "ld" column to int64
        # expected_result['ld'] = expected_result['ld'].astype('int32')

        pd.testing.assert_frame_equal(expected_result, actual_result)

    def test_summaryp_example_2(self):
        """
        Test the summaryp function with multiple reference and hypothesis strings.

        This test evaluates the SUMMARYP function with multiple reference and hypothesis text sequences.
        It verifies that the calculated SUMMARYP dataframe aligns with the expected output dataframe.

        """
        ref = "the tower caused minor discontent because it blocked sight lines of central park"
        hyp = "the tower caused minor discontent because it blocked sightlines of central park"

        # Generate the actual_result DataFrame
        actual_result = summaryp(
            ref,
            hyp,
            insertions_weight=0.5,
            deletions_weight=0.5,
            substitutions_weight=1,
        )

        data = {
            "wer": [0.15384615384615385],
            "werp": [0.11538461538461539],
            "ld": [2],
            "m": [13],
            "insertions": [0],
            "deletions": [1],
            "substitutions": [1],
            "inserted_words": [[]],
            "deleted_words": [["sight"]],
            "substituted_words": [[("lines", "sightlines")]],
        }

        expected_result = pd.DataFrame(data)

        pd.testing.assert_frame_equal(expected_result, actual_result)

    def test_summaryp_example_3(self):
        """
        Test the summaryp function with numerical reference and hypothesis inputs.

        This test evaluates the SUMMARYP function with numerical reference and hypothesis inputs.
        It verifies that the numerical input will raise an AttributeError.

        """
        ref = [1, 2, 3, 4]
        hyp = [2, 3, 3, 3]
        # The actual return value is None from the try/except block in wer module
        expected_result = None

        self.assertEqual(summaryp(ref, hyp, 0.5, 0.5, 1), expected_result)

    def test_summaryp_insertion_only(self):
        """
        Test the summaryp function with a hypothesis that adds one word to the reference.

        It verifies that the added word is counted and listed as an insertion and that the insertion weight is
        applied to the weighted Word Error Rate.
        """
        actual_result = summaryp("the cat sat", "the big cat sat", 0.5, 0.5, 1)

        expected_result = pd.DataFrame(
            {
                "wer": [0.3333333333333333],
                "werp": [0.16666666666666666],
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

    def test_summaryp_deletion_only(self):
        """
        Test the summaryp function with a hypothesis that omits one word of the reference.

        It verifies that the omitted word is counted and listed as a deletion and that the deletion weight is
        applied to the weighted Word Error Rate.
        """
        actual_result = summaryp(["the big cat sat"], ["the cat sat"], 0.5, 0.5, 1)

        expected_result = pd.DataFrame(
            {
                "wer": [0.25],
                "werp": [0.125],
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

    def test_summaryp_blank_reference_in_list(self):
        """
        Test the summaryp function with a list containing a blank reference.

        It verifies that the function prints a message naming the index of the blank reference and returns None.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = summaryp(["", "a"], ["x", "a"])

        self.assertIsNone(result)
        self.assertEqual(
            buffer.getvalue().strip(),
            "ZeroDivisionError: Invalid input: reference must not be blank. A blank reference was found at index 0.",
        )

    def test_summaryp_empty_sequences(self):
        """
        Test the summaryp function with two empty lists.

        It verifies that two empty lists yield a dataframe with the ten result columns and no rows.
        """
        actual_result = summaryp([], [])

        expected_columns = [
            "wer",
            "werp",
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
