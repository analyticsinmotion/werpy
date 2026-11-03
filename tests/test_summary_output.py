# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_summary_output.py

This module contains a set of unit tests for the output argument of the 'summary' and 'summaryp' functions in the
'werpy' package, and for these functions when pandas or polars cannot be imported. pandas and polars are optional
dependencies of werpy.

With output="pandas", the default, the two functions return a pandas DataFrame. With output="polars", they return a
polars DataFrame. With output="records", they return a list with one dictionary for each pair of texts, whose keys are
the columns of the DataFrame. This form works with a plain install of werpy.

To run the tests, execute this module as the main program.
"""

import importlib
import subprocess  # nosec B404 - the test starts the interpreter that runs it, with fixed arguments
import sys
import unittest
from unittest import mock

import numpy as np
import werpy
from werpy.exceptions import WerpyError, WerpyTypeError, WerpyValueError

try:
    import pandas as pd
except ImportError:  # pandas is an optional dependency of werpy
    pd = None

try:
    import polars as pl
except ImportError:  # polars is an optional dependency of werpy
    pl = None

OUTPUT = importlib.import_module("werpy._output")
PANDAS = unittest.skipIf(pd is None, "pandas is not installed")
POLARS = unittest.skipIf(pl is None, "polars is not installed")
SUMMARY_COLUMNS = ["wer", "ld", "m", "insertions", "deletions", "substitutions", "inserted_words", "deleted_words",
                   "substituted_words"]
SUMMARYP_COLUMNS = ["wer", "werp", *SUMMARY_COLUMNS[1:]]
OUTPUT_MESSAGE = 'output must be "pandas", "polars" or "records", not {}.'
REF = ["it is consumed domestically", "the cat sat", "rufino street in makati", "the big cat sat"]
HYP = ["it is consumed domestically", "the big cat sat", "rofino street in mccauti", "the cat sat"]
WEIGHT_SETS = ((1, 1, 1), (0.5, 0.5, 1), (2.5, 0.75, 1.3), (0, 0, 0))


def missing_pandas_message(function):
    """Return the message of the ImportError for output="pandas" when pandas cannot be imported."""
    return (
        f"{function}() returns a pandas DataFrame, and pandas could not be imported. Install pandas, for example "
        f'with pip install "werpy[pandas]", or call {function}() with output="polars" to get a polars DataFrame or '
        'with output="records" to get a list of dictionaries.'
    )


def missing_polars_message(function):
    """Return the message of the ImportError for output="polars" when polars cannot be imported."""
    return (
        f'{function}() with output="polars" returns a polars DataFrame, and polars could not be imported. Install '
        f'polars, for example with pip install "werpy[polars]", or call {function}() with output="records" to get a '
        "list of dictionaries."
    )


def polars_schema(columns):
    """Return the types of the columns of the polars DataFrame of summary() or summaryp(), in column order."""
    words = pl.List(pl.String)
    types = {
        "wer": pl.Float64,
        "werp": pl.Float64,
        "inserted_words": words,
        "deleted_words": words,
        "substituted_words": pl.List(pl.Struct({"reference": pl.String, "hypothesis": pl.String})),
    }
    return [(name, types.get(name, pl.Int64)) for name in columns]


def as_polars_rows(records):
    """Return the records with each substitution as the dictionary that polars gives for a Struct value."""
    return [
        {**record, "substituted_words": [{"reference": ref, "hypothesis": hyp}
                                         for ref, hyp in record["substituted_words"]]}
        for record in records
    ]


def batch_forms():
    """Return the batch forms of REF and HYP: lists, tuples, numpy arrays of dtype U and object, and empty lists."""
    return [
        (REF, HYP),
        (tuple(REF), tuple(HYP)),
        (np.array(REF), np.array(HYP)),
        (np.array(REF, dtype=object), np.array(HYP, dtype=object)),
        ([], []),
    ]


class TestSummaryOutput(unittest.TestCase):
    """
    This class contains unit tests for the output argument of the summary and summaryp functions.
    """

    def test_summary_records(self):
        """
        Test that summary with output="records" gives one dictionary for each pair, with the nine keys in the order
        of the columns, a float rate, int counts and lists of words, for a pair of strings and for a batch.
        """
        self.assertEqual(
            werpy.summary("this is a test", "this is the test", output="records"),
            [{"wer": 0.25, "ld": 1, "m": 4, "insertions": 0, "deletions": 0, "substitutions": 1, "inserted_words": [],
              "deleted_words": [], "substituted_words": [("a", "the")]}],
        )
        records = werpy.summary(REF, HYP, output="records")
        self.assertEqual(len(records), len(REF))
        self.assertEqual(records[1], {"wer": 1 / 3, "ld": 1, "m": 3, "insertions": 1, "deletions": 0,
                                      "substitutions": 0, "inserted_words": ["big"], "deleted_words": [],
                                      "substituted_words": []})
        for record in records:
            self.assertEqual(list(record), SUMMARY_COLUMNS)
            self.assertEqual([type(value) for value in record.values()], [float, *[int] * 5, list, list, list])

    def test_summaryp_records(self):
        """
        Test that summaryp with output="records" gives one dictionary for each pair, with the ten keys in the order
        of the columns, float rates, int counts and lists of words, for a pair of strings and for a batch.
        """
        self.assertEqual(
            werpy.summaryp("the cat sat", "the big cat sat", 0.5, 0.5, 1, output="records"),
            [{"wer": 1 / 3, "werp": 1 / 6, "ld": 1, "m": 3, "insertions": 1, "deletions": 0, "substitutions": 0,
              "inserted_words": ["big"], "deleted_words": [], "substituted_words": []}],
        )
        records = werpy.summaryp(REF, HYP, 0.5, 0.5, 1, output="records")
        self.assertEqual([record["werp"] for record in records], werpy.werps(REF, HYP, 0.5, 0.5, 1))
        for record in records:
            self.assertEqual(list(record), SUMMARYP_COLUMNS)
            self.assertEqual([type(value) for value in record.values()], [float, float, *[int] * 5, list, list, list])

    def test_records_of_empty_batches(self):
        """
        Test that two empty lists, tuples or numpy arrays give an empty list with output="records".
        """
        for reference, hypothesis in (([], []), ((), ()), (np.array([], dtype=str), np.array([], dtype=str))):
            with self.subTest(form=type(reference).__name__):
                self.assertEqual(werpy.summary(reference, hypothesis, output="records"), [])
                self.assertEqual(werpy.summaryp(reference, hypothesis, output="records"), [])

    @PANDAS
    def test_records_equal_the_dataframes(self):
        """
        Test that the records of summary and summaryp hold the values of the rows of their DataFrames, and that a
        DataFrame of the records, with the columns of the DataFrame, equals it, for every batch form, for a pair of
        strings and for weight sets of each direction.
        """
        forms = [*batch_forms(), (REF[2], HYP[2])]
        for reference, hypothesis in forms:
            with self.subTest(function="summary", form=type(reference).__name__, size=len(reference)):
                frame = werpy.summary(reference, hypothesis)
                records = werpy.summary(reference, hypothesis, output="records")
                self.assertEqual(records, frame.to_dict("records"))
                pd.testing.assert_frame_equal(pd.DataFrame(records, columns=frame.columns), frame)
            for weights in WEIGHT_SETS:
                with self.subTest(function="summaryp", form=type(reference).__name__, weights=weights):
                    frame = werpy.summaryp(reference, hypothesis, *weights)
                    records = werpy.summaryp(reference, hypothesis, *weights, output="records")
                    self.assertEqual(records, frame.to_dict("records"))
                    expected = pd.DataFrame(records, columns=frame.columns)
                    if not records:
                        # An empty DataFrame of summaryp has a float64 werp column and object columns.
                        expected = expected.astype({"werp": "float64"})
                    pd.testing.assert_frame_equal(expected, frame)

    @PANDAS
    def test_default_is_the_dataframe(self):
        """
        Test that output="pandas" gives the DataFrame that the default gives.
        """
        pd.testing.assert_frame_equal(werpy.summary(REF, HYP, output="pandas"), werpy.summary(REF, HYP))
        pd.testing.assert_frame_equal(
            werpy.summaryp(REF, HYP, 0.5, 0.5, 1, output="pandas"), werpy.summaryp(REF, HYP, 0.5, 0.5, 1)
        )

    @POLARS
    def test_summary_polars(self):
        """
        Test that summary with output="polars" gives a polars DataFrame with the nine columns in order and their
        types, one row for each pair, and the values of the records, each substitution as a Struct with the fields
        reference and hypothesis, for every batch form and for a pair of strings.
        """
        for reference, hypothesis in [*batch_forms()[:4], (REF[2], HYP[2])]:
            with self.subTest(form=type(reference).__name__):
                frame = werpy.summary(reference, hypothesis, output="polars")
                self.assertIsInstance(frame, pl.DataFrame)
                self.assertEqual(list(frame.schema.items()), polars_schema(SUMMARY_COLUMNS))
                records = werpy.summary(reference, hypothesis, output="records")
                self.assertEqual(frame.height, len(records))
                self.assertEqual(frame.to_dicts(), as_polars_rows(records))
        frame = werpy.summary(REF[2], HYP[2], output="polars")
        self.assertEqual(frame["substituted_words"].to_list(),
                         [[{"reference": "rufino", "hypothesis": "rofino"},
                           {"reference": "makati", "hypothesis": "mccauti"}]])

    @POLARS
    def test_summaryp_polars(self):
        """
        Test that summaryp with output="polars" gives a polars DataFrame with the ten columns in order and their
        types, and the values of the records, for every batch form, for a pair of strings and for weight sets of each
        direction. Its werp column equals werps().
        """
        for reference, hypothesis in [*batch_forms()[:4], (REF[2], HYP[2])]:
            for weights in WEIGHT_SETS:
                with self.subTest(form=type(reference).__name__, weights=weights):
                    frame = werpy.summaryp(reference, hypothesis, *weights, output="polars")
                    self.assertIsInstance(frame, pl.DataFrame)
                    self.assertEqual(list(frame.schema.items()), polars_schema(SUMMARYP_COLUMNS))
                    records = werpy.summaryp(reference, hypothesis, *weights, output="records")
                    self.assertEqual(frame.to_dicts(), as_polars_rows(records))
                    self.assertEqual(frame["werp"].to_list(), werpy.werps(reference, hypothesis, *weights))

    @POLARS
    def test_polars_of_empty_batches(self):
        """
        Test that two empty lists, tuples or numpy arrays give a polars DataFrame with no rows and the columns and
        types of a batch with output="polars".
        """
        for reference, hypothesis in (([], []), ((), ()), (np.array([], dtype=str), np.array([], dtype=str))):
            with self.subTest(form=type(reference).__name__):
                frame = werpy.summary(reference, hypothesis, output="polars")
                self.assertEqual(frame.shape, (0, 9))
                self.assertEqual(list(frame.schema.items()), polars_schema(SUMMARY_COLUMNS))
                frame = werpy.summaryp(reference, hypothesis, output="polars")
                self.assertEqual(frame.shape, (0, 10))
                self.assertEqual(list(frame.schema.items()), polars_schema(SUMMARYP_COLUMNS))

    @POLARS
    def test_polars_substitutions_equal_the_records(self):
        """
        Test that the substituted_words column of the polars DataFrame of summary and summaryp equals the column that
        polars builds from the pairs of the records, and that explode and unnest give the columns reference and
        hypothesis with the pairs of the records in order, for a batch and for a pair of strings.
        """
        substitutions = pl.List(pl.Struct({"reference": pl.String, "hypothesis": pl.String}))
        for function, args in ((werpy.summary, ()), (werpy.summaryp, (2.5, 0.75, 1.3))):
            for form, reference, hypothesis in (("batch", REF, HYP), ("pair", REF[2], HYP[2])):
                with self.subTest(function=function.__name__, form=form):
                    frame = function(reference, hypothesis, *args, output="polars")
                    records = function(reference, hypothesis, *args, output="records")
                    expected = pl.Series(
                        "substituted_words", [record["substituted_words"] for record in records], dtype=substitutions
                    )
                    self.assertTrue(frame["substituted_words"].equals(expected, check_dtypes=True, check_names=True))
                    pairs = frame.select(pl.col("substituted_words").explode()).drop_nulls().unnest("substituted_words")
                    self.assertEqual(pairs.columns, ["reference", "hypothesis"])
                    self.assertEqual(pairs.rows(), [pair for record in records for pair in record["substituted_words"]])

    def test_output_errors(self):
        """
        Test that an output that is not a string raises a WerpyTypeError, and a string other than "pandas", "polars"
        and "records" a WerpyValueError, with messages that name the value, in summary and summaryp, and that the
        reference, the hypothesis and the weights are checked before output.
        """
        cases = [
            (None, WerpyTypeError, OUTPUT_MESSAGE.format("NoneType")),
            (1, WerpyTypeError, OUTPUT_MESSAGE.format("int")),
            (b"pandas", WerpyTypeError, OUTPUT_MESSAGE.format("bytes")),
            (["records"], WerpyTypeError, OUTPUT_MESSAGE.format("list")),
            ("dataframe", WerpyValueError, OUTPUT_MESSAGE.format('"dataframe"')),
            ("Pandas", WerpyValueError, OUTPUT_MESSAGE.format('"Pandas"')),
            ("Polars", WerpyValueError, OUTPUT_MESSAGE.format('"Polars"')),
            ("", WerpyValueError, OUTPUT_MESSAGE.format('""')),
        ]
        for function in (werpy.summary, werpy.summaryp):
            for output, error, message in cases:
                with self.subTest(function=function.__name__, output=repr(output)):
                    with self.assertRaises(error) as context:
                        function("a b", "a c", output=output)
                    self.assertIs(type(context.exception), error)
                    self.assertEqual(str(context.exception), message)
            with self.subTest(function=function.__name__, check="the inputs first"):
                with self.assertRaises(WerpyValueError) as context:
                    function("", "a c", output=None)
                self.assertIn("reference must not be blank", str(context.exception))
        with self.assertRaises(WerpyTypeError) as context:
            werpy.summaryp("a b", "a c", "x", output="dataframe")
        self.assertIn("insertions_weight", str(context.exception))
        with self.assertRaises(TypeError):
            werpy.summary("a b", "a c", "records")  # pylint: disable=too-many-function-args

    def test_without_pandas(self):
        """
        Test that, when pandas cannot be imported, summary and summaryp raise an ImportError for output="pandas" whose
        message names the function, the pandas extra and the other outputs and whose cause is the error of the import
        of pandas, that output="records" and, with polars, output="polars" still give their results, and that a
        reference, a hypothesis, a weight or an output that is not valid raises its WerpyError first.
        """
        cause = ModuleNotFoundError("No module named 'pandas'", name="pandas")
        expected = werpy.summary(REF, HYP, output="records")
        expected_p = werpy.summaryp(REF, HYP, 0.5, 0.5, 1, output="records")
        with mock.patch.object(OUTPUT, "pd", None), mock.patch.object(OUTPUT, "PANDAS_ERROR", cause):
            for function, args in ((werpy.summary, ()), (werpy.summaryp, (0.5, 0.5, 1))):
                for kwargs in ({}, {"output": "pandas"}):
                    with self.subTest(function=function.__name__, kwargs=kwargs):
                        with self.assertRaises(ImportError) as context:
                            function(REF, HYP, *args, **kwargs)
                        self.assertIs(type(context.exception), ImportError)
                        self.assertNotIsInstance(context.exception, WerpyError)
                        self.assertEqual(str(context.exception), missing_pandas_message(function.__name__))
                        self.assertEqual(context.exception.name, "pandas")
                        self.assertIs(context.exception.__cause__, cause)
            self.assertEqual(werpy.summary(REF, HYP, output="records"), expected)
            self.assertEqual(werpy.summaryp(REF, HYP, 0.5, 0.5, 1, output="records"), expected_p)
            if pl is not None:
                self.assertEqual(werpy.summary(REF, HYP, output="polars").to_dicts(), as_polars_rows(expected))
            invalid = [
                (werpy.summary, ("", "a b"), {}),
                (werpy.summary, ("a b", "a c"), {"output": "x"}),
                (werpy.summaryp, ("a b", "a c", -1), {}),
                (werpy.summaryp, (["a b"], "a c"), {}),
            ]
            for function, args, kwargs in invalid:
                with self.subTest(function=function.__name__, args=repr(args), kwargs=kwargs):
                    with self.assertRaises(WerpyError):
                        function(*args, **kwargs)

    def test_without_polars(self):
        """
        Test that, when polars cannot be imported, summary and summaryp raise an ImportError for output="polars" whose
        message names the function, the polars extra and output="records" and whose cause is the error of the import
        of polars, that output="records" and, with pandas, the default still give their results, and that a
        reference, a hypothesis, a weight or an output that is not valid raises its WerpyError first.
        """
        expected = werpy.summary(REF, HYP, output="records")
        with mock.patch.dict(sys.modules, {"polars": None}):  # an import of polars now raises ModuleNotFoundError
            for function, args in ((werpy.summary, ()), (werpy.summaryp, (0.5, 0.5, 1))):
                with self.subTest(function=function.__name__):
                    with self.assertRaises(ImportError) as context:
                        function(REF, HYP, *args, output="polars")
                    self.assertIs(type(context.exception), ImportError)
                    self.assertNotIsInstance(context.exception, WerpyError)
                    self.assertEqual(str(context.exception), missing_polars_message(function.__name__))
                    self.assertEqual(context.exception.name, "polars")
                    self.assertIs(type(context.exception.__cause__), ModuleNotFoundError)
            self.assertEqual(werpy.summary(REF, HYP, output="records"), expected)
            if pd is not None:
                self.assertEqual(werpy.summary(REF, HYP).to_dict("records"), expected)
            invalid = [
                (werpy.summary, ("", "a b"), {"output": "polars"}),
                (werpy.summary, ("a b", "a c"), {"output": "Polars"}),
                (werpy.summaryp, ("a b", "a c", -1), {"output": "polars"}),
                (werpy.summaryp, (["a b"], "a c"), {"output": "polars"}),
            ]
            for function, args, kwargs in invalid:
                with self.subTest(function=function.__name__, args=repr(args), kwargs=kwargs):
                    with self.assertRaises(WerpyError):
                        function(*args, **kwargs)

    def test_import_when_pandas_cannot_be_imported(self):
        """
        Test that werpy._output imports when pandas cannot be imported: pandas is then None, the error of the import
        is kept, and summary raises the ImportError for output="pandas" with that error as its cause. The module is
        imported again afterwards with pandas.
        """
        saved = sys.modules.get("pandas")
        sys.modules["pandas"] = None  # an import of pandas now raises ModuleNotFoundError
        try:
            importlib.reload(OUTPUT)
            self.assertIsNone(OUTPUT.pd)
            self.assertIsInstance(OUTPUT.PANDAS_ERROR, ModuleNotFoundError)
            with self.assertRaises(ImportError) as context:
                werpy.summary("a b", "a c")
            self.assertIs(context.exception.__cause__, OUTPUT.PANDAS_ERROR)
            self.assertEqual(len(werpy.summary("a b", "a c", output="records")), 1)
        finally:
            if saved is None:
                del sys.modules["pandas"]
            else:
                sys.modules["pandas"] = saved
            importlib.reload(OUTPUT)
        self.assertIs(OUTPUT.pd, pd)
        self.assertIs(OUTPUT.PANDAS_ERROR is None, pd is not None)

    def test_polars_imported_on_request(self):
        """
        Test, in a new process, that import werpy does not import polars, and that output="polars" imports it when
        polars is installed.
        """
        code = (
            "import sys\n"
            "import werpy\n"
            "print('polars' in sys.modules)\n"
            "try:\n"
            "    werpy.summary('a b', 'a c', output='polars')\n"
            "except ImportError:\n"
            "    pass\n"
            "print('polars' in sys.modules)\n"
        )
        result = subprocess.run(  # nosec B603 - the interpreter that runs this test, with fixed arguments
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["False", str(pl is not None)])

    def test_new_process_without_pandas_and_polars(self):
        """
        Test, in a new process in which neither pandas nor polars can be imported, that werpy imports, that the five
        other public functions work, that summary and summaryp give records, and that summary raises the ImportError
        for its default and for output="polars".
        """
        code = (
            "import sys\n"
            "sys.modules['pandas'] = None\n"
            "sys.modules['polars'] = None\n"
            "import werpy\n"
            "assert werpy.wer('a b', 'a c') == 0.5 and werpy.wers(['a b'], ['a c']) == [0.5]\n"
            "assert werpy.werp('a b', 'a c') == 0.5 and werpy.werps('a b', 'a c') == [0.5]\n"
            "assert werpy.normalize('A b.') == 'a b'\n"
            "assert werpy.summary('a b', 'a c', output='records')[0]['ld'] == 1\n"
            "assert werpy.summaryp('a b', 'a c', output='records')[0]['werp'] == 0.5\n"
            "for output in ('pandas', 'polars'):\n"
            "    try:\n"
            "        werpy.summary('a b', 'a c', output=output)\n"
            "    except ImportError as error:\n"
            "        print(type(error).__name__, error.name, type(error.__cause__).__name__)\n"
        )
        result = subprocess.run(  # nosec B603 - the interpreter that runs this test, with fixed arguments
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split("\n")[:2],
                         ["ImportError pandas ModuleNotFoundError", "ImportError polars ModuleNotFoundError"])

    def test_records_example(self):
        """
        Test the list of dictionaries example of the README (Example 9) and of the summarization page: the records
        that summary gives with output="records" print as the documented line.
        """
        ref = ["it was beautiful and sunny today"]
        hyp = ["it was a beautiful and sunny day"]
        self.assertEqual(
            str(werpy.summary(ref, hyp, output="records")),
            "[{'wer': 0.3333333333333333, 'ld': 2, 'm': 6, 'insertions': 1, 'deletions': 0, 'substitutions': 1, "
            "'inserted_words': ['a'], 'deleted_words': [], 'substituted_words': [('today', 'day')]}]",
        )

    @POLARS
    def test_polars_example(self):
        """
        Test the polars DataFrame example of the README (Example 10) and of the summarization page: the four selected
        columns of the polars DataFrame of summary hold the documented rows, with the documented types.
        """
        ref = ["it was beautiful and sunny today", "she cited multiple reasons why"]
        hyp = ["it was a beautiful and sunny day", "she sighted multiple reasons why"]
        summary = werpy.summary(ref, hyp, output="polars")
        frame = summary.select("wer", "ld", "inserted_words", "substituted_words")
        types = dict(polars_schema(SUMMARY_COLUMNS))
        self.assertEqual(list(frame.schema.items()),
                         [(name, types[name]) for name in ("wer", "ld", "inserted_words", "substituted_words")])
        self.assertEqual(
            frame.rows(),
            [
                (1 / 3, 2, ["a"], [{"reference": "today", "hypothesis": "day"}]),
                (0.2, 1, [], [{"reference": "cited", "hypothesis": "sighted"}]),
            ],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
