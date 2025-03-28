# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_normalize.py

This module contains unit tests for the 'normalize' function in the 'werpy.normalize' module. 
The 'normalize' function is responsible for normalizing translations, 
both reference and hypothesis, to facilitate word error rate (WER) calculations.

Tested functionality:
- Normalization of reference translations.
- Normalization of hypothesis translations.

Test cases:
- 'test_normalize_reference_translation': Test the normalization of reference translations.
- 'test_normalize_hypothesis_translation': Test the normalization of hypothesis translations.

Each test case compares the output of the 'normalize' function with the expected normalized translations 
and raises an AssertionError if the actual output does not match the expected values.

To run the tests, execute this module as the main program.

Note: If the 'normalize' module is not imported successfully, 
an ImportError is raised to ensure that the required module is available for testing.
"""

import unittest
from werpy.normalize import normalize


class TestNormalize(unittest.TestCase):
    """
    TestNormalize - Unit tests for the 'normalize' function in 'werpy.normalize'.

    This class defines test cases to verify the correctness of the 'normalize' function,
    which is responsible for normalizing translations, both reference and hypothesis,
    to enable word error rate (WER) calculations.

    Test cases:
    - 'test_normalize_reference_translation': Test the normalization of reference translations.
    - 'test_normalize_hypothesis_translation': Test the normalization of hypothesis translations.
    - 'test_normalize_string': Test the normalization of a single string.

    Each test case compares the output of the 'normalize' function with the expected normalized translations
    and raises an AssertionError if the actual output does not match the expected values.

    Attributes:
        None

    Methods:
        - 'test_normalize_reference_translation': Test the normalization of reference translations.
        - 'test_normalize_hypothesis_translation': Test the normalization of hypothesis translations.
        - 'test_normalize_string': Test the normalization of a single string.
        - 'test_normalize_invalid_types': Test the normalize function with various invalid input types.

    To run the tests, execute this class as part of the test suite in the main program.

    Note: If the 'normalize' module is not imported successfully,
    an ImportError is raised to ensure that the required module is available for testing.
    """

    def test_normalize_reference(self):
        """
        Test the normalization of reference translations.
        """
        reference = [
            "     It is consumed domestically           and exported to other countries.     ",
            "The Sugar Bear character was popular enough to have occasional premium toys.",
            "It is one of the most watched television networks in the country.",
            "It could be carried and prepared by the individual soldier.",
            "He was executed in a Lubyanka prison cellar.",
            "Rufino Street in Makati, right inside the Makati Central Business District.",
            "Its estuary is considered to have abnormally low rates of dissolved oxygen.",
            "He later cited his first wife Anita as the inspiration for the song.",
            "Gadya is the nearest rural locality.",
            "Taxes are a tool in the adjustment of the economy.",
        ]
        expected_normalized_reference = [
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

        self.assertEqual(normalize(reference), expected_normalized_reference)

    def test_normalize_hypothesis(self):
        """
        Test the normalization of hypothesis translations.
        """
        hypothesis = [
            "it is consumed domestically and exported to other countries ",
            "the sugar bare character was popular enough to have occasional premium toys ",
            "it is one of the most watched television networks in the country ",
            "it could be carried and prepared by the individual soldier ",
            "he was executed in alabianca prison seller ",
            "rofino street in mccauti right inside the macasi central business district ",
            "it's estiary is considered to have a normally low rates of dissolved oxygen ",
            "he later sighted his first wife anita as the inspiration for the song ",
            "gadia is the nearest rural locality ",
            "taxes are a tool in the adjustment of the economy ",
        ]
        expected_normalized_hypothesis = [
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

        self.assertEqual(normalize(hypothesis), expected_normalized_hypothesis)

    def test_normalize_string(self):
        """
        Test the normalization of a single string.
        """
        reference = normalize(
            " it's Consumed Domestically  And exported to other countries."
        )

        expected_normalized_reference = (
            "its consumed domestically and exported to other countries"
        )

        self.assertEqual(normalize(reference), expected_normalized_reference)

    def test_normalize_integers(self):
        """
        Test the normalize function with numerical reference and hypothesis inputs.

        This test evaluates the NORMALIZE function with numerical reference and hypothesis inputs.
        It verifies that the numerical input will raise an TypeError.

        """
        ref = [1, 2, 3, 4]
        with self.assertRaises(TypeError):
            normalize(ref)

    def test_normalize_invalid_types(self):
        """
        Test the normalize function with various invalid input types.

        This test evaluates the normalize function with inputs from the following types:
           - int
           - float
           - bool
           - range
           - dict
           - bytes
           - bytearray
           - complex

        It verifies that the function raises a TypeError for these inputs.
        """
        invalid_inputs = [
            42,  # int
            3.14,  # float
            True,  # bool
            range(10),  # range
            {"key": "value"},  # dict
            b"bytes",  # bytes
            bytearray(b"bytearray"),  # bytearray
            complex(1, 1)  # complex
        ]

        for invalid_input in invalid_inputs:
            with self.assertRaises(TypeError):
                normalize(invalid_input)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
