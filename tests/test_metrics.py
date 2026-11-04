# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_metrics.py

This module contains a set of unit tests for the 'metrics', 'metrics_fast' and 'metrics_wer_only' functions
of the compiled 'werpy._metrics' module in the 'werpy' package.

Each of these functions accepts a pair of strings or a pair of lists, as the 'error_handler' function gives
them. For a pair of lists it returns a list with one row, a tuple, for each reference and hypothesis pair. The
tests in this module check the rows for lists of equal length, and check that lists with different numbers
of elements raise a ValueError with the message that the 'error_handler' function raises for the same input,
which it raises for numpy arrays too.

To run the tests, execute this module as the main program.

For more details on the 'werpy._metrics' module, please refer to the 'werpy' package documentation.
"""

import unittest
from werpy._errorhandler import error_handler
from werpy._metrics import metrics, metrics_fast, metrics_wer_only  # pylint: disable=no-name-in-module


LENGTH_MESSAGE = "The Reference and Hypothesis input parameters must have the same number of elements."
ROUTERS = (metrics, metrics_fast, metrics_wer_only)

try:
    import numpy as np
except ImportError:  # NumPy is not a dependency of werpy
    np = None

NUMPY = unittest.skipIf(np is None, "NumPy is not installed")


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
        Test that each function returns a list with one row, a tuple, for each pair of two lists of equal length, and
        that each row is the row of the pair given as two strings.
        """
        reference = ["i love cold pizza", "the sugar bear character was popular"]
        hypothesis = ["i love pizza", "the sugar bare character was popular"]
        for router, width in ((metrics, 9), (metrics_fast, 6), (metrics_wer_only, 3)):
            rows = router(reference, hypothesis)
            self.assertIs(type(rows), list, msg=router.__name__)
            self.assertEqual([(type(row), len(row)) for row in rows], [(tuple, width)] * 2, msg=router.__name__)
            self.assertEqual(rows, [router(ref, hyp) for ref, hyp in zip(reference, hypothesis)], msg=router.__name__)

    def test_metrics_empty_sequences(self):
        """
        Test that each function returns an empty list for two empty lists.
        """
        for router in ROUTERS:
            self.assertEqual(router([], []), [], msg=router.__name__)

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

    @NUMPY
    def test_metrics_numpy_arrays_of_different_lengths(self):
        """
        Test that numpy arrays of string and object dtype with different numbers of elements, with the longer array in
        either position, raise the length ValueError in error_handler, which gives the routers their lists.
        """
        for dtype in (str, object):
            longer = np.array(["i love cold pizza", "the sugar bear"], dtype=dtype)
            shorter = np.array(["i love pizza"], dtype=dtype)
            self.assert_length_error(error_handler, longer, shorter)
            self.assert_length_error(error_handler, shorter, longer)

    @NUMPY
    def test_metrics_list_and_numpy_array_of_different_lengths(self):
        """
        Test that a list and a numpy array with different numbers of elements, with the list in either position, raise
        the length ValueError in error_handler, which gives the routers their lists.
        """
        self.assert_length_error(error_handler, ["i love cold pizza", "the sugar bear"], np.array(["i love pizza"]))
        self.assert_length_error(error_handler, np.array(["i love cold pizza"]), ["i love pizza", "the sugar bare"])

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
