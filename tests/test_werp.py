# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_werp.py

This module contains a set of unit tests for the 'werp' function in the 'werpy' package.

The 'werp' function calculates a weighted Word Error Rate (WER) across multiple text sequence pairs. 
It takes as input a reference text and a hypothesis text, both as sequences of words. 
Additionally, the user can specify custom weights or penalties for insertion, deletion, and substitution errors. 
These weights allow users to penalize certain error types more than others when calculating WER. 
The 'werp' function returns a single weighted WER value aggregated across all the input sequences, 
unlike 'wers' which provides per-sequence WERs. This collective weighted WER metric allows understanding 
model performance with custom emphasis on different error categories across the entire test set.

To run the tests, execute this module as the main program.

For more details on the 'werp' function and how to use it, please refer to the 'werpy' package documentation.

Note: If the 'werp' module is not imported successfully, an ImportError is raised 
to ensure that the required module is available for testing.
"""

import unittest
from werpy.werp import werp


class TestWerp(unittest.TestCase):
    """
    This class contains unit tests for the 'werp' function, which calculates the weighted Word Error Rate (WER)
    between reference and hypothesis text sequences.
    """

    def test_werp_example_1(self):
        """
        Test the werp function with a simple example.

        This test checks the WERP function with a basic example of reference and hypothesis strings
        and ensures that the calculated WERP matches the expected result.
        """
        self.assertEqual(werp("i love cold pizza", "i love pizza", 0.5, 0.5, 1), 0.125)

    def test_werp_example_2(self):
        """
        Test the werp function with multiple reference and hypothesis strings.

        This test evaluates the WERP function with multiple reference and hypothesis text sequences.
        It verifies that the calculated WERP aligns with the expected results.

        """
        ref = ["i love cold pizza", "the sugar bear character was popular"]
        hyp = ["i love pizza", "the sugar bare character was popular"]
        expected_result = 0.15

        self.assertEqual(
            werp(
                ref,
                hyp,
                insertions_weight=0.5,
                deletions_weight=0.5,
                substitutions_weight=1,
            ),
            expected_result,
        )

    def test_werp_example_3(self):
        """
        Test the werp function with multiple reference and hypothesis strings.

        This test evaluates the WERP function with multiple reference and hypothesis text sequences.
        It verifies that the calculated WERP aligns with the expected results.

        """
        ref = ["no one else could claim that", "she cited multiple reasons why"]
        hyp = ["no one else could claim that", "she sighted multiple reasons why"]
        expected_result = 0.09090909090909091

        self.assertEqual(werp(ref, hyp, 0.5, 0.5, 1), expected_result)

    def test_werp_example_4(self):
        """
        Test the werp function with multiple reference and hypothesis strings.

        This test evaluates the WERP function with multiple reference and hypothesis text sequences.
        It verifies that the calculated WERP aligns with the expected results.

        """
        ref = ["it was beautiful and sunny today"]
        hyp = ["it was a beautiful and sunny day"]
        expected_result = 0.25

        self.assertEqual(werp(ref, hyp, 0.5, 0.5, 1), expected_result)

    def test_werp_example_5(self):
        """
        Test the werp function with longer text sequences.

        This test assesses the WERP function's performance with longer reference and hypothesis text sequences.
        It verifies that the calculated WERP closely matches the expected results.

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
        expected_result = 0.10679611650485436

        self.assertEqual(werp(ref, hyp, 0.5, 0.5, 1), expected_result)

    def test_werp_example_6(self):
        """
        Test the werp function with numerical reference and hypothesis inputs.

        This test evaluates the WERP function with numerical reference and hypothesis inputs.
        It verifies that the numerical input will raise an AttributeError.

        """
        ref = [1, 2, 3, 4]
        hyp = [2, 3, 3, 3]
        # The actual return value is None from the try/except block in werp module
        expected_result = None

        self.assertEqual(werp(ref, hyp, 0.5, 0.5, 1), expected_result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
