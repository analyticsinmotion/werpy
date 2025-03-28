# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_werps.py

This module contains a set of unit tests for the 'werps' function in the 'werpy' package.

The 'werps' function calculates a weighted Word Error Rate (WER) for each text sequence pair. 
It takes as input a reference text and a hypothesis text, both as sequences of words. 
Additionally, the user can specify custom weights or penalties for insertion, deletion, 
and substitution errors. These weights allow users to penalize certain error types more 
than others when calculating the per-sequence WERs. Unlike 'werp' which provides an aggregated 
weighted WER across all sequences, 'werps' returns the weighted WER for each individual sequence pair. 
By providing per-sequence weighted WERs, 'werps' enables a more detailed, granular error analysis 
that accounts for custom error penalties and weights on a per-example basis.

To run the tests, execute this module as the main program.

For more details on the 'werps' function and how to use it, please refer to the 'werpy' package documentation.

Note: If the 'werps' module is not imported successfully, an ImportError is raised 
to ensure that the required module is available for testing.
"""

import unittest
from werpy.werps import werps


class TestWerps(unittest.TestCase):
    """
    This class contains unit tests for the 'werps' function, which calculates the weighted Word Error Rate (WER)
    between reference and hypothesis text sequences.
    """

    def test_werps_example_1(self):
        """
        Test the werps function with a simple example.

        This test checks the WERPS function with a basic example of reference and hypothesis strings
        and ensures that the calculated WERPS matches the expected result.
        """
        self.assertEqual(werps("i love cold pizza", "i love pizza", 0.5, 0.5, 1), 0.125)

    def test_werps_example_2(self):
        """
        Test the werps function with multiple reference and hypothesis strings.

        This test evaluates the WERPS function with multiple reference and hypothesis text sequences.
        It verifies that the calculated WERPS aligns with the expected results.

        """
        ref = ["i love cold pizza", "the sugar bear character was popular"]
        hyp = ["i love pizza", "the sugar bare character was popular"]
        expected_result = [0.125, 0.16666666666666666]

        self.assertEqual(
            werps(
                ref,
                hyp,
                insertions_weight=0.5,
                deletions_weight=0.5,
                substitutions_weight=1,
            ),
            expected_result,
        )

    def test_werps_example_3(self):
        """
        Test the werps function with multiple reference and hypothesis strings.

        This test evaluates the WERPS function with multiple reference and hypothesis text sequences.
        It verifies that the calculated WERPS aligns with the expected results.

        """
        ref = ["no one else could claim that", "she cited multiple reasons why"]
        hyp = ["no one else could claim that", "she sighted multiple reasons why"]
        expected_result = [0.0, 0.2]

        self.assertEqual(werps(ref, hyp, 0.5, 0.5, 1), expected_result)

    def test_werps_example_4(self):
        """
        Test the werps function with multiple reference and hypothesis strings.

        This test evaluates the WERPS function with multiple reference and hypothesis text sequences.
        It verifies that the calculated WERPS aligns with the expected results.

        """
        ref = ["it was beautiful and sunny today"]
        hyp = ["it was a beautiful and sunny day"]
        expected_result = [0.25]

        self.assertEqual(werps(ref, hyp, 0.5, 0.5, 1), expected_result)

    def test_werps_example_5(self):
        """
        Test the werps function with multiple reference and hypothesis strings.

        This test evaluates the WERPS function with multiple reference and hypothesis text sequences.
        It verifies that the calculated WERPS aligns with the expected results.

        """
        ref = [
            "it blocked sight lines of central park",
            "her father was an alderman in the city government",
        ]
        hyp = [
            "it blocked sightlines of central park",
            "our father was an elder man in the city government",
        ]
        expected_result = [0.21428571428571427, 0.2777777777777778]

        self.assertEqual(werps(ref, hyp, 0.5, 0.5, 1), expected_result)

    def test_werps_example_6(self):
        """
        Test the werps function with longer text sequences.

        This test assesses the WERPS function's performance with longer reference and hypothesis text sequences.
        It verifies that the calculated WERPS closely matches the expected results.

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
            0.3125,
            0.2727272727272727,
            0.20833333333333334,
            0.07692307692307693,
            0.16666666666666666,
            0.0,
        ]

        self.assertEqual(werps(ref, hyp, 0.5, 0.5, 1), expected_result)

    def test_werps_example_7(self):
        """
        Test the werps function with numerical reference and hypothesis inputs.

        This test evaluates the WERPS function with numerical reference and hypothesis inputs.
        It verifies that the numerical input will raise an AttributeError.

        """
        ref = [1, 2, 3, 4]
        hyp = [2, 3, 3, 3]
        # The actual return value is None from the try/except block in werps module
        expected_result = None

        self.assertEqual(werps(ref, hyp, 0.5, 0.5, 1), expected_result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
