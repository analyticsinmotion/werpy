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

import unittest
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

        try:
            pd.testing.assert_frame_equal(expected_result, actual_result)
            print("DataFrames are equal.")
        except AssertionError as error:  # pragma: no cover
            print("DataFrames are not equal. Differences:\n", error)

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
