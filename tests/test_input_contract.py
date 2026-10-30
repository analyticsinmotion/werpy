# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_input_contract.py

This module contains a set of unit tests for the input contract of the 'werpy' package: the inputs that the
'normalize', 'wer', 'wers', 'werp', 'werps', 'summary' and 'summaryp' functions accept, the exceptions and messages
that they raise for an input that is not valid, and the order of the checks.

A text is a string. A batch is a list, a tuple or a one-dimensional numpy array whose elements are all strings. The six
metric functions take a reference and a hypothesis of the same kind, and two batches must have the same number of
elements. A reference must not be blank. The weights of 'werp', 'werps' and 'summaryp' are real numbers that are
finite and at least 0. An input that breaks one of these rules raises 'werpy.exceptions.WerpyTypeError' or
'werpy.exceptions.WerpyValueError'.

To run the tests, execute this module as the main program.
"""

import decimal
import fractions
import os
import tempfile
import unittest
import warnings
import numpy as np
import pandas as pd
import werpy
from werpy.exceptions import WerpyError, WerpyTypeError, WerpyValueError


SIX = (werpy.wer, werpy.wers, werpy.werp, werpy.werps, werpy.summary, werpy.summaryp)
WEIGHTED = (werpy.werp, werpy.werps, werpy.summaryp)
WEIGHT_NAMES = ("insertions_weight", "deletions_weight", "substitutions_weight")
KIND_MESSAGE = "Reference and hypothesis must both be strings, or both be lists, tuples or numpy arrays."
LENGTH_MESSAGE = "The Reference and Hypothesis input parameters must have the same number of elements."
EMPTY_MESSAGE = (
    "Reference and hypothesis must not be empty. The Word Error Rate is not defined for empty sequences."
)
# One deletion, one substitution and one insertion, with no tied alignment: each weight changes the result.
PAIR = ("a b c d e", "b x d e f")
BATCH = (["a b c d e", "a b c"], ["b x d e f", "a b c d"])
FILE_LINES = (
    "It's very popular in Antarctica.",
    "  The Sugar Bear character  ",
    "",
    "Taxes are a TOOL, in the economy!",
)
FILE_LINES_NORMALIZED = [
    "its very popular in antarctica",
    "the sugar bear character",
    "",
    "taxes are a tool in the economy",
]


class Text(str):
    """A str subclass."""


def type_message(name, value):
    """Return the message for a reference, hypothesis or text of the wrong type."""
    return f"{name} must be a str, or a list, tuple or one-dimensional numpy array of str, not {type(value).__name__}."


def dimensions_message(name, ndim):
    """Return the message for a numpy array that does not have exactly one dimension."""
    return f"{name} must be a one-dimensional numpy array, not an array with {ndim} dimensions."


def element_message(name, element, index):
    """Return the message for an element of a list, tuple or numpy array that is not a string."""
    return (
        f"Every element of {name} must be a str. An element of type {type(element).__name__} was found at index "
        f"{index}."
    )


def weight_type_message(name, weight):
    """Return the message for a weight that is not a real number."""
    return f"{name} must be a real number, such as an int or a float, not {type(weight).__name__}."


def weight_value_message(name):
    """Return the message for a weight that is negative, NaN, infinite or too large for a float."""
    return f"{name} must be a finite number of at least 0."


def call(function, reference, hypothesis, weights=()):
    """Call one of the six metric functions. The weights are passed to werp, werps and summaryp only."""
    if function in WEIGHTED:
        return function(reference, hypothesis, *weights)
    return function(reference, hypothesis)


def object_array(items):
    """Return a one-dimensional numpy array of dtype object that holds the given items as they are."""
    array = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        array[index] = item
    return array


def string_dtype_array(texts):
    """Return a numpy array of StringDType that holds the texts, or None when NumPy has no StringDType."""
    string_dtype = getattr(getattr(np, "dtypes", None), "StringDType", None)
    return None if string_dtype is None else np.array(texts, dtype=string_dtype())


def generator(texts):
    """Return a generator of the texts."""
    return (text for text in texts)


def wrong_type_values():
    """Return new values of types that are neither a text nor a batch. A generator can be used only once."""
    return [
        None,
        5,
        2.5,
        b"a b",
        bytearray(b"a b"),
        memoryview(b"a b"),
        pd.Series(["a b"]),
        {"a b"},
        generator(["a b"]),
        {"a b": 1},
        {"a b": 1}.keys(),
        range(2),
    ]


class TestInputContract(unittest.TestCase):
    """
    This class contains unit tests for the input contract of the werpy functions.
    """

    def _assert_raises(self, error, message, function, *args):
        """
        Assert that function(*args) raises exactly the class 'error' with the message 'message', and return the
        exception.
        """
        with self.assertRaises(error) as context:
            function(*args)
        self.assertIs(type(context.exception), error)
        self.assertEqual(str(context.exception), message)
        return context.exception

    def _assert_same_result(self, actual, expected):
        """
        Assert that two results of a metric function are equal: data frames value by value, and other results with
        the same type and value.
        """
        if isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected)
        else:
            self.assertIs(type(actual), type(expected))
            self.assertEqual(actual, expected)

    def test_type_message(self):
        """
        Test that a reference or hypothesis of the wrong type raises a WerpyTypeError whose message names the argument
        and the type, in the six metric functions, and that normalize raises it for a text of the wrong type.
        """
        for function in SIX:
            for value, other in zip(wrong_type_values(), wrong_type_values()):
                with self.subTest(function=function.__name__, type=type(value).__name__):
                    self._assert_raises(WerpyTypeError, type_message("reference", value), call, function, value, "a b")
                    self._assert_raises(WerpyTypeError, type_message("hypothesis", other), call, function, "a b", other)
        for value in wrong_type_values():
            with self.subTest(function="normalize", type=type(value).__name__):
                self._assert_raises(WerpyTypeError, type_message("text", value), werpy.normalize, value)

    def test_dimensions(self):
        """
        Test that a numpy array with no dimensions, with two dimensions or with three dimensions, and a numpy matrix,
        raise a WerpyTypeError whose message names the argument and the number of dimensions, for each argument of the
        six metric functions and for the text of normalize.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            matrix = np.asmatrix(np.array(["a b", "c d"]))
        arrays = [np.array("a b"), np.array([["a b"], ["c d"]]), np.array([[["a b"]]]), matrix]
        for array in arrays:
            for function in SIX:
                with self.subTest(function=function.__name__, type=type(array).__name__, ndim=array.ndim):
                    self._assert_raises(
                        WerpyTypeError, dimensions_message("reference", array.ndim), call, function, array, ["a b"]
                    )
                    self._assert_raises(
                        WerpyTypeError, dimensions_message("hypothesis", array.ndim), call, function, ["a b"], array
                    )
            with self.subTest(function="normalize", type=type(array).__name__, ndim=array.ndim):
                self._assert_raises(WerpyTypeError, dimensions_message("text", array.ndim), werpy.normalize, array)

    def test_kind_message(self):
        """
        Test that a string paired with a list, a tuple, a numpy array or an empty numpy array, in either order, raises
        a WerpyTypeError with the kind message in the six metric functions. The arrays include one whose only element
        equals the string.
        """
        batches = [
            ["a b c"],
            ("a b c",),
            np.array(["a b c"]),
            np.array(["a b c"], dtype=object),
            np.array(["a x c"]),
            np.array([], dtype=str),
        ]
        for batch in batches:
            for function in SIX:
                with self.subTest(function=function.__name__, batch=repr(batch)):
                    self._assert_raises(WerpyTypeError, KIND_MESSAGE, call, function, "a b c", batch)
                    self._assert_raises(WerpyTypeError, KIND_MESSAGE, call, function, batch, "a b c")

    def test_tuples_accepted(self):
        """
        Test that tuples, and tuples mixed with lists and numpy arrays, give the results of the same texts given as
        lists, in the six metric functions.
        """
        reference, hypothesis = ["a b c", "d e"], ["a x c", "d e f"]
        weights = (0.5, 0.5, 1)
        expected = {function: call(function, reference, hypothesis, weights) for function in SIX}
        forms = [
            (tuple(reference), tuple(hypothesis)),
            (tuple(reference), hypothesis),
            (reference, tuple(hypothesis)),
            (tuple(reference), np.array(hypothesis)),
            (np.array(reference, dtype=object), tuple(hypothesis)),
        ]
        for ref, hyp in forms:
            for function in SIX:
                with self.subTest(function=function.__name__, forms=(type(ref).__name__, type(hyp).__name__)):
                    self._assert_same_result(call(function, ref, hyp, weights), expected[function])

    def test_element_types(self):
        """
        Test that an element of a list that is not a string raises a WerpyTypeError whose message names the argument,
        the type of the element and its index, for each argument of the six metric functions and for the text of
        normalize.
        """
        elements = [None, 1, 1.5, True, b"a b", ["a b"], ("a b",), np.int64(1), np.bytes_(b"a b")]
        for element in elements:
            for function in SIX:
                with self.subTest(function=function.__name__, element=type(element).__name__):
                    self._assert_raises(
                        WerpyTypeError,
                        element_message("reference", element, 1),
                        call,
                        function,
                        ["a b", element],
                        ["a b", "c d"],
                    )
                    self._assert_raises(
                        WerpyTypeError,
                        element_message("hypothesis", element, 1),
                        call,
                        function,
                        ["a b", "c d"],
                        ["a b", element],
                    )
            with self.subTest(function="normalize", element=type(element).__name__):
                self._assert_raises(
                    WerpyTypeError, element_message("text", element, 1), werpy.normalize, ["a b", element]
                )

    def test_d4_through_elements(self):
        """
        Test that a hypothesis element that is a numpy array or a pandas Series equal to the reference element raises
        a WerpyTypeError whose message names the element type, in the six metric functions, for a list and for a numpy
        array of dtype object.
        """
        cases = [
            (["a b c"], [np.array(["a b c"])]),
            (["a b c"], [np.array("a b c")]),
            (["a b c"], [pd.Series(["a b c"])]),
            (object_array(["a b c"]), object_array([np.array(["a b c"])])),
        ]
        for reference, hypothesis in cases:
            for function in SIX:
                with self.subTest(function=function.__name__, element=type(hypothesis[0]).__name__):
                    self._assert_raises(
                        WerpyTypeError,
                        element_message("hypothesis", hypothesis[0], 0),
                        call,
                        function,
                        reference,
                        hypothesis,
                    )

    def test_blank_bytes_reference(self):
        """
        Test that a blank bytes element of the reference raises a WerpyTypeError with the element message in the six
        metric functions, before the blank reference check.
        """
        for function in SIX:
            with self.subTest(function=function.__name__):
                self._assert_raises(
                    WerpyTypeError, element_message("reference", b"   ", 0), call, function, [b"   "], [b"a b"]
                )

    def test_empty_batches(self):
        """
        Test that two empty lists, tuples or numpy arrays raise a WerpyValueError with the empty message in wer and
        werp, give an empty list in wers and werps, and give a data frame with all its columns and no rows in summary
        and summaryp.
        """
        forms = [
            ([], []),
            ((), ()),
            (np.array([], dtype=str), np.array([], dtype=str)),
            (np.array([]), np.array([])),
        ]
        for reference, hypothesis in forms:
            form = f"{type(reference).__name__} {getattr(reference, 'dtype', '')}".strip()
            for function in (werpy.wer, werpy.werp):
                with self.subTest(function=function.__name__, form=form):
                    self._assert_raises(WerpyValueError, EMPTY_MESSAGE, call, function, reference, hypothesis)
            for function in (werpy.wers, werpy.werps):
                with self.subTest(function=function.__name__, form=form):
                    self.assertEqual(call(function, reference, hypothesis), [])
            for function, columns in ((werpy.summary, 9), (werpy.summaryp, 10)):
                with self.subTest(function=function.__name__, form=form):
                    frame = call(function, reference, hypothesis)
                    self.assertIsInstance(frame, pd.DataFrame)
                    self.assertEqual(frame.shape, (0, columns))

    def test_order_of_checks(self):
        """
        Test that an input with two faults raises the exception of the first check that fails: the type and the
        dimensions of the reference, then of the hypothesis, the kind, the number of elements, empty batches, the
        elements, blank references, and the weights.
        """
        cases = [
            (werpy.wer, (None, 5), WerpyTypeError, type_message("reference", None)),
            (werpy.wer, (np.array("a"), None), WerpyTypeError, dimensions_message("reference", 0)),
            (werpy.wer, (["a", None], ["a"]), WerpyValueError, LENGTH_MESSAGE),
            (werpy.wer, ([""], [None]), WerpyTypeError, element_message("hypothesis", None, 0)),
            (werpy.werp, (None, "a", -1), WerpyTypeError, type_message("reference", None)),
            (werpy.werp, ([], [], -1), WerpyValueError, EMPTY_MESSAGE),
            (werpy.werp, ("a b", "a c", "x", -1), WerpyTypeError, weight_type_message("insertions_weight", "x")),
        ]
        for function, args, error, message in cases:
            with self.subTest(function=function.__name__, args=repr(args)):
                self._assert_raises(error, message, function, *args)

    def test_valid_inputs_unchanged(self):
        """
        Test that a str subclass, numpy.str_, a list of numpy.str_, a numpy array of dtype object, a list mixed with a
        numpy array, and a numpy array of StringDType give the results of the same texts given as strings or lists, in
        the six metric functions. NumPy 1.x has no StringDType, so that case is skipped there.
        """
        reference, hypothesis = ["a b c", "d e"], ["a x c", "d e f"]
        weights = (0.5, 0.5, 1)
        expected_text = {function: call(function, reference[0], hypothesis[0], weights) for function in SIX}
        expected_batch = {function: call(function, reference, hypothesis, weights) for function in SIX}
        text_forms = [(Text(reference[0]), Text(hypothesis[0])), (np.str_(reference[0]), np.str_(hypothesis[0]))]
        batch_forms = [
            ([np.str_(text) for text in reference], [np.str_(text) for text in hypothesis]),
            (np.array(reference, dtype=object), np.array(hypothesis, dtype=object)),
            (reference, np.array(hypothesis)),
            (np.array(reference), hypothesis),
        ]
        for forms, expected in ((text_forms, expected_text), (batch_forms, expected_batch)):
            for ref, hyp in forms:
                for function in SIX:
                    with self.subTest(function=function.__name__, reference=repr(ref)):
                        self._assert_same_result(call(function, ref, hyp, weights), expected[function])
        with self.subTest(form="StringDType"):
            ref, hyp = string_dtype_array(reference), string_dtype_array(hypothesis)
            if ref is None:
                self.skipTest("NumPy 1.x has no StringDType.")
            for function in SIX:
                self._assert_same_result(call(function, ref, hyp, weights), expected_batch[function])

    def test_weight_types(self):
        """
        Test that a weight that is not a real number raises a WerpyTypeError whose message names the weight and its
        type, in each position, in werp, werps and summaryp.
        """
        weights = [
            True,
            np.bool_(True),
            "0.5",
            None,
            [1, 2],
            (1, 2),
            np.array([1, 2]),
            np.array(0.5),
            decimal.Decimal("0.5"),
            complex(1, 0),
            np.timedelta64(1, "s"),
        ]
        for weight in weights:
            for position, name in enumerate(WEIGHT_NAMES):
                args = [1, 1, 1]
                args[position] = weight
                for function in WEIGHTED:
                    with self.subTest(function=function.__name__, weight=repr(weight), position=name):
                        self._assert_raises(WerpyTypeError, weight_type_message(name, weight), function, *PAIR, *args)

    def test_weight_values(self):
        """
        Test that a weight that is negative, NaN, infinite or too large for a float raises a WerpyValueError whose
        message names the weight, in each position, in werp, werps and summaryp.
        """
        for weight in (-1, -0.5, float("nan"), float("inf"), float("-inf"), 10**400):
            for position, name in enumerate(WEIGHT_NAMES):
                args = [1, 1, 1]
                args[position] = weight
                for function in WEIGHTED:
                    with self.subTest(function=function.__name__, weight=repr(weight)[:20], position=name):
                        self._assert_raises(WerpyValueError, weight_value_message(name), function, *PAIR, *args)

    def test_weights_accepted(self):
        """
        Test that weights given as an int, a float, a numpy integer, a numpy floating-point number or a fraction give
        the result of the same weights given as floats, and that the weights 0 and -0.0 are valid, in werp, werps and
        summaryp, for a pair of strings and for a batch.
        """
        for weight in (2, 0.5, np.int64(2), np.uint8(2), np.float32(0.5), fractions.Fraction(1, 2), 0, -0.0):
            number = float(weight)
            for function in WEIGHTED:
                for reference, hypothesis in (PAIR, BATCH):
                    with self.subTest(function=function.__name__, weight=repr(weight), reference=repr(reference)):
                        self._assert_same_result(
                            function(reference, hypothesis, weight, weight, weight),
                            function(reference, hypothesis, number, number, number),
                        )

    def test_all_zero_weights(self):
        """
        Test that three weights of 0 give a weighted Word Error Rate of 0.0 in werp, werps and summaryp.
        """
        for zero in (0, 0.0):
            with self.subTest(weight=repr(zero)):
                self.assertEqual(werpy.werp(*PAIR, zero, zero, zero), 0.0)
                self.assertEqual(werpy.werp(*BATCH, zero, zero, zero), 0.0)
                self.assertEqual(werpy.werps(*PAIR, zero, zero, zero), [0.0])
                self.assertEqual(werpy.werps(*BATCH, zero, zero, zero), [0.0, 0.0])
                self.assertEqual(werpy.summaryp(*PAIR, zero, zero, zero)["werp"].tolist(), [0.0])
                self.assertEqual(werpy.summaryp(*BATCH, zero, zero, zero)["werp"].tolist(), [0.0, 0.0])

    def test_weights_not_broadcast(self):
        """
        Test that a list, a tuple or a numpy array that holds one weight for each pair of a batch raises a
        WerpyTypeError with the weight type message in werp, werps and summaryp, so that no weight is applied pair by
        pair.
        """
        for weights in ([1, 2], (1, 2), np.array([1, 2])):
            for function in WEIGHTED:
                with self.subTest(function=function.__name__, weights=repr(weights)):
                    self._assert_raises(
                        WerpyTypeError, weight_type_message("insertions_weight", weights), function, *BATCH, weights
                    )

    def test_normalize_contract(self):
        """
        Test the input contract of normalize. A string, a list, a tuple and a one-dimensional numpy array of strings
        are accepted. A pandas Series, a generator, a set and a dict view raise a WerpyTypeError with the type message,
        a numpy array with two dimensions raises the dimension message, and an element that is not a string raises
        the element message. Each of these exceptions is also a TypeError. The lines of a file, given as a list of the
        lines without their line ends, are normalized line by line.
        """
        texts = ["It's very popular in Antarctica.", "  The Sugar Bear character  "]
        expected = ["its very popular in antarctica", "the sugar bear character"]
        self.assertEqual(werpy.normalize(texts[0]), expected[0])
        for form in (texts, tuple(texts), np.array(texts), np.array(texts, dtype=object)):
            with self.subTest(form=type(form).__name__):
                self.assertEqual(werpy.normalize(form), expected)
        for value in (pd.Series(texts), generator(texts), set(texts), {"a": texts[0]}.values(), {texts[0]: 1}.keys()):
            with self.subTest(value=type(value).__name__):
                error = self._assert_raises(WerpyTypeError, type_message("text", value), werpy.normalize, value)
                self.assertIsInstance(error, TypeError)
        error = self._assert_raises(WerpyTypeError, dimensions_message("text", 2), werpy.normalize, np.array([texts]))
        self.assertIsInstance(error, TypeError)
        error = self._assert_raises(WerpyTypeError, element_message("text", 1, 1), werpy.normalize, [texts[0], 1])
        self.assertIsInstance(error, TypeError)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "transcripts.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("".join(line + "\n" for line in FILE_LINES))
            with open(path, encoding="utf-8") as f:
                normalized_lines = werpy.normalize([line.rstrip("\n") for line in f])
        self.assertEqual(normalized_lines, FILE_LINES_NORMALIZED)

    def test_every_invalid_input_is_a_werpy_error(self):
        """
        Test that each exception that the seven functions raise for an input that is not valid is a WerpyError.
        """
        pairs = [
            (None, "a b"),
            ("a b", 5),
            (np.array("a b"), np.array("a x")),
            (np.array([["a b"]]), np.array([["a b"]])),
            ("a b", ["a b"]),
            (["a b", "c d"], ["a b"]),
            (("a b", "c d"), ("a b",)),
            ([1], [1]),
            ([b"a b"], [b"a b"]),
            (["a b"], [np.array(["a b"])]),
            (["   "], ["a b"]),
            ("", "a b"),
            (pd.Series(["a b"]), pd.Series(["a b"])),
        ]
        for reference, hypothesis in pairs:
            for function in SIX:
                with self.subTest(function=function.__name__, reference=repr(reference)):
                    with self.assertRaises(WerpyError):
                        call(function, reference, hypothesis)
        for function in (werpy.wer, werpy.werp):
            with self.subTest(function=function.__name__, reference="[]"):
                with self.assertRaises(WerpyError):
                    call(function, [], [])
        for weight in ("0.5", None, True, -1, float("nan"), 10**400, [1, 2]):
            for function in WEIGHTED:
                with self.subTest(function=function.__name__, weight=repr(weight)[:20]):
                    with self.assertRaises(WerpyError):
                        function(*BATCH, weight)
        for text in (None, 42, np.array("a b"), np.array([["a b"]]), [1], pd.Series(["a b"]), ["a b", None]):
            with self.subTest(function="normalize", text=repr(text)):
                with self.assertRaises(WerpyError):
                    werpy.normalize(text)

    def test_except_clauses_still_catch(self):
        """
        Test that an except clause for the built-in exception that werpy 3.x raised for an input still catches the
        exception for that input, and test the two documented changes in all six metric functions: tuples of
        different lengths raise a WerpyValueError, which is not a TypeError, and two-dimensional numpy arrays of
        different lengths raise a WerpyTypeError, which is not a ValueError.
        """
        cases = [
            (werpy.wer, (np.array([["a b"], ["c d"]]), np.array([["a b"], ["c x"]])), AttributeError),
            (werpy.wer, (np.array("a b c"), np.array("a x c")), TypeError),
            (werpy.wer, ([None], ["a b"]), AttributeError),
            (werpy.wer, ([1, 2], [1, 2]), AttributeError),
            (werpy.wer, ([["a b"]], [["a b"]]), AttributeError),
            (werpy.werp, ("a b", "a c", "0.5"), TypeError),
            (werpy.werp, ("a b", "a c", None), TypeError),
            (werpy.normalize, (None,), TypeError),
            (werpy.normalize, ([1, 2],), TypeError),
        ]
        for function, args, legacy_error in cases:
            with self.subTest(function=function.__name__, args=repr(args)):
                with self.assertRaises(legacy_error):
                    function(*args)
        for function in SIX:
            with self.subTest(function=function.__name__, change="tuples of different lengths"):
                error = self._assert_raises(WerpyValueError, LENGTH_MESSAGE, call, function, ("a b", "c d"), ("a b",))
                self.assertNotIsInstance(error, TypeError)
            with self.subTest(function=function.__name__, change="two-dimensional arrays of different lengths"):
                error = self._assert_raises(
                    WerpyTypeError,
                    dimensions_message("reference", 2),
                    call,
                    function,
                    np.array([["a b"], ["c d"]]),
                    np.array([["a b"]]),
                )
                self.assertNotIsInstance(error, ValueError)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
