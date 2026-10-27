# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_metrics.py

This module contains a set of unit tests for the 'metrics', 'metrics_fast' and 'metrics_wer_only' functions
of the compiled 'werpy.metrics' module in the 'werpy' package.

Each of these functions accepts a pair of strings or a pair of lists or numpy arrays. For a pair of
sequences it returns a two-dimensional array with one row per reference and hypothesis pair. The tests in
this module check the shape of the result for sequences of equal length, and check that sequences with
different numbers of elements raise a ValueError with the message that the 'error_handler' function raises
for the same input.

To run the tests, execute this module as the main program.

For more details on the 'werpy.metrics' module, please refer to the 'werpy' package documentation.
"""

import unittest
import numpy as np
from werpy.errorhandler import error_handler
from werpy.metrics import metrics, metrics_fast, metrics_wer_only


LENGTH_MESSAGE = "The Reference and Hypothesis input parameters must have the same number of elements."
ROUTERS = (metrics, metrics_fast, metrics_wer_only)


class TestMetrics(unittest.TestCase):
    """
    This class contains unit tests for the 'metrics', 'metrics_fast' and 'metrics_wer_only' functions, which
    route a pair of strings to a single calculation and a pair of sequences to a batch calculation.
    """

    def assert_length_error(self, router, reference, hypothesis):
        """
        Assert that the function raises a ValueError with the length message for the given inputs.
        """
        with self.assertRaises(ValueError, msg=router.__name__) as context:
            router(reference, hypothesis)
        self.assertEqual(str(context.exception), LENGTH_MESSAGE, msg=router.__name__)

    def test_metrics_equal_lengths(self):
        """
        Test that each function returns one row per pair for two lists of equal length.
        """
        reference = ["i love cold pizza", "the sugar bear character was popular"]
        hypothesis = ["i love pizza", "the sugar bare character was popular"]
        self.assertEqual(metrics(reference, hypothesis).shape, (2, 9))
        self.assertEqual(metrics_fast(reference, hypothesis).shape, (2, 6))
        self.assertEqual(metrics_wer_only(reference, hypothesis).shape, (2, 3))

    def test_metrics_empty_sequences(self):
        """
        Test that each function returns an array with no rows for two empty lists.
        """
        self.assertEqual(metrics([], []).shape, (0, 9))
        self.assertEqual(metrics_fast([], []).shape, (0, 6))
        self.assertEqual(metrics_wer_only([], []).shape, (0, 3))

    def test_metrics_more_references_than_hypotheses(self):
        """
        Test that metrics raises a ValueError when the references have more elements than the hypotheses.
        """
        self.assert_length_error(metrics, ["i love cold pizza", "the sugar bear"], ["i love pizza"])

    def test_metrics_more_hypotheses_than_references(self):
        """
        Test that metrics raises a ValueError when the hypotheses have more elements than the references.
        """
        self.assert_length_error(metrics, ["i love cold pizza"], ["i love pizza", "the sugar bare"])

    def test_metrics_empty_reference_sequence(self):
        """
        Test that metrics raises a ValueError for an empty list of references and a list of one hypothesis.
        """
        self.assert_length_error(metrics, [], ["i love pizza"])

    def test_metrics_empty_hypothesis_sequence(self):
        """
        Test that metrics raises a ValueError for a list of one reference and an empty list of hypotheses.
        """
        self.assert_length_error(metrics, ["i love cold pizza"], [])

    def test_metrics_numpy_arrays_of_different_lengths(self):
        """
        Test that metrics raises a ValueError for numpy arrays of string and object dtype with different
        numbers of elements, with the longer array in either position.
        """
        for dtype in (str, object):
            longer = np.array(["i love cold pizza", "the sugar bear"], dtype=dtype)
            shorter = np.array(["i love pizza"], dtype=dtype)
            self.assert_length_error(metrics, longer, shorter)
            self.assert_length_error(metrics, shorter, longer)

    def test_metrics_list_and_numpy_array_of_different_lengths(self):
        """
        Test that metrics raises a ValueError for a list and a numpy array with different numbers of elements,
        with the list in either position.
        """
        self.assert_length_error(metrics, ["i love cold pizza", "the sugar bear"], np.array(["i love pizza"]))
        self.assert_length_error(metrics, np.array(["i love cold pizza"]), ["i love pizza", "the sugar bare"])

    def test_metrics_fast_different_lengths(self):
        """
        Test that metrics_fast raises a ValueError for two lists with different numbers of elements, with the
        longer list in either position.
        """
        self.assert_length_error(metrics_fast, ["i love cold pizza", "the sugar bear"], ["i love pizza"])
        self.assert_length_error(metrics_fast, ["i love cold pizza"], ["i love pizza", "the sugar bare"])

    def test_metrics_wer_only_different_lengths(self):
        """
        Test that metrics_wer_only raises a ValueError for two lists with different numbers of elements, with
        the longer list in either position.
        """
        self.assert_length_error(metrics_wer_only, ["i love cold pizza", "the sugar bear"], ["i love pizza"])
        self.assert_length_error(metrics_wer_only, ["i love cold pizza"], ["i love pizza", "the sugar bare"])

    def test_routers_long_sequences_of_different_lengths(self):
        """
        Test that each function raises a ValueError for lists of 1,000 and 999 elements, in either position.
        """
        long_list = [f"word {index} of the sequence" for index in range(1000)]
        for router in ROUTERS:
            self.assert_length_error(router, long_list, long_list[:999])
            self.assert_length_error(router, long_list[:999], long_list)

    def test_routers_invalid_elements_and_different_lengths(self):
        """
        Test that each function raises the length ValueError, and not an error for an element, when the lists
        have different numbers of elements and contain elements that are not strings.
        """
        for router in ROUTERS:
            self.assert_length_error(router, [None, "a b"], [None])

    def test_routers_length_message_matches_error_handler(self):
        """
        Test that the message each function raises for sequences of different lengths is the message that
        error_handler raises for the same input.
        """
        reference = ["i love cold pizza", "the sugar bear"]
        hypothesis = ["i love pizza"]
        with self.assertRaises(ValueError) as expected:
            error_handler(reference, hypothesis)
        for router in ROUTERS:
            with self.assertRaises(ValueError, msg=router.__name__) as context:
                router(reference, hypothesis)
            self.assertEqual(str(context.exception), str(expected.exception), msg=router.__name__)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
