# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_weighted_alignment.py

This module contains a set of unit tests for the alignment on which the 'werp', 'werps' and 'summaryp' functions of the
'werpy' package count the errors.

These functions count the errors on an alignment of the reference and hypothesis words with the fewest edits. When
several such alignments exist, they use the one with the lowest weighted cost: the one with the fewest substitutions
when a substitution weighs more than the mean of an insertion and a deletion, and the one with the most substitutions
when it weighs less. When a substitution weighs exactly that mean, they use the alignment of 'summary'.

The tests compare the three functions with the exhaustive search of 'alignment_reference.py' over every alignment of
short texts. They also check the documented examples, the batches, the rates of weights so large that the float
arithmetic overflows, the exact comparison of the weights in 'substitution_order', the compiled kernels of
'werpy._weighted', and that its copies of the token mapping and the length check of 'werpy._metrics' give the same
results.

To run the tests, execute this module as the main program.
"""

import importlib
import importlib.util
import itertools
import random
import sys
import math
import unittest
import warnings
from fractions import Fraction
import alignment_reference as oracle
import werpy
from werpy._errorhandler import substitution_order

try:
    import numpy as np
except ImportError:  # NumPy is not a dependency of werpy
    np = None

try:
    import polars as pl
except ImportError:  # polars is an optional dependency of werpy
    pl = None


KERNELS = importlib.import_module("werpy._weighted")
METRICS = importlib.import_module("werpy._metrics")
# The tests that read the DataFrames of summary and summaryp need pandas, an optional dependency of werpy.
PANDAS = unittest.skipIf(importlib.util.find_spec("pandas") is None, "pandas is not installed")
# One weight set of each direction: a substitution heavier than the mean, lighter than the mean, and exactly the mean.
DIRECTION_WEIGHTS = {1: (0.5, 0.5, 1), -1: (2.5, 0.75, 1.3), 0: (1, 3, 2)}
EQUAL_MEAN_WEIGHTS = ((1, 1, 1), (2, 2, 2), (0, 0, 0), (1, 3, 2), (0.5, 0.5, 0.5))
ALL_WEIGHTS = (*DIRECTION_WEIGHTS.values(), *EQUAL_MEAN_WEIGHTS, (0.1, 0.3, 0.7), (0, 0, 1), (1, 1, 0))
SWAPPED = ("please call me back tomorrow", "please call back me tomorrow")
LIGHTER = ("a a b c", "b c b")
# Pairs with more than one alignment with the fewest edits; the last has a single count of each kind among them.
TIE_PAIRS = (
    ("a b", "b a"),
    ("a x", "x b"),
    LIGHTER,
    ("b b c a", "c a c"),
    SWAPPED,
    ("the cat sat on the mat", "cat the sat on the mat"),
    ("a b c", "c a b"),
)
ORDER_MESSAGE = "The order must be 1 or -1."
LENGTH_MESSAGE = "The Reference and Hypothesis input parameters must have the same number of elements."
# Words for the token mapping: equal strings, and strings that differ only in case, in their Unicode code points, in one
# character, by a NUL or a lone surrogate, or at the end of a long word.
TOKEN_WORDS = ("w", "W", "é", "é", "ß", "ss", "1", "01", "a\x00b", "a\x00c", "\U0001f600", "\ud800",
               "x" * 1000, "x" * 999 + "y")


def texts(max_words):
    """Return every text of 0 to max_words words over the words a, b and c."""
    return [" ".join(words) for count in range(max_words + 1) for words in itertools.product("abc", repeat=count)]


def exhaustive_pairs(max_words):
    """Return every pair of texts of 0 to max_words words over a, b and c whose reference is not blank."""
    return [(ref, hyp) for ref in texts(max_words) for hyp in texts(max_words) if ref]


def sample_pairs(count, seed):
    """
    Return count pairs of random texts over 2 to 5 distinct words, with 1 to 6 reference words, 0 to 6 hypothesis
    words and at most 10 words in all.
    """
    rng = random.Random(seed)
    pairs = []
    for _ in range(count):
        words = "abcde"[: rng.randint(2, 5)]
        ref_count = rng.randint(1, 6)
        hyp_count = min(rng.randint(0, 6), 10 - ref_count)
        ref = " ".join(rng.choice(words) for _ in range(ref_count))
        hyp = " ".join(rng.choice(words) for _ in range(hyp_count))
        pairs.append((ref, hyp))
    return pairs


def summaryp_row(frame, index=0):
    """Return one row of a summaryp frame as Python values, in the order of its columns."""
    row = frame.iloc[index]
    return [
        float(row["wer"]),
        float(row["werp"]),
        int(row["ld"]),
        int(row["m"]),
        int(row["insertions"]),
        int(row["deletions"]),
        int(row["substitutions"]),
        list(row["inserted_words"]),
        list(row["deleted_words"]),
        list(row["substituted_words"]),
    ]


def counts_of(row):
    """Return the counts and the word lists of a row of a summary or summaryp frame, as a Series or a record."""
    return [
        int(row["insertions"]),
        int(row["deletions"]),
        int(row["substitutions"]),
        list(row["inserted_words"]),
        list(row["deleted_words"]),
        list(row["substituted_words"]),
    ]


def summary_counts(frame, index=0):
    """Return the counts and the word lists of one row of a summary or summaryp frame."""
    return counts_of(frame.iloc[index])


def outcome(function, *args):
    """Return the class and message of the exception that function(*args) raises, or None when it returns."""
    try:
        function(*args)
    except ValueError as exc:
        return type(exc), str(exc)
    return None


def werp_columns(reference, hypothesis, weights):
    """
    Return the werp column of summaryp in each form that the environment can build: the records, and the pandas and
    polars DataFrames when pandas and polars can be imported.
    """
    records = werpy.summaryp(reference, hypothesis, *weights, output="records")
    columns = {"records": [record["werp"] for record in records]}
    if importlib.util.find_spec("pandas") is not None:
        columns["pandas"] = werpy.summaryp(reference, hypothesis, *weights)["werp"].tolist()
    if pl is not None:
        columns["polars"] = werpy.summaryp(reference, hypothesis, *weights, output="polars")["werp"].to_list()
    return columns


def nearest(result, exact):
    """
    Return True when a finite float result is the float nearest to an exact Fraction of at least the smallest normal
    float: no float is nearer, and of two floats that are equally near, the result is the one whose last binary digit
    is 0. Above the largest float, 2**1024 stands for the next float.
    """
    above = math.nextafter(result, math.inf)
    neighbours = (Fraction(math.nextafter(result, 0.0)), Fraction(2) ** 1024 if above == math.inf else Fraction(above))
    distance = abs(Fraction(result) - exact)
    odd = int(math.frexp(result)[0] * 2**53) % 2 == 1
    return all(abs(neighbour - exact) > distance or (abs(neighbour - exact) == distance and not odd)
               for neighbour in neighbours)


def expected_row(ref_text, hyp_text, weights, paths=None):
    """
    Return the summaryp row that the exhaustive search gives for a pair of texts and weights. The werp column is the
    rate of 'werp' on a pair.
    """
    ref, hyp = ref_text.split(), hyp_text.split()
    path = oracle.choose(ref, hyp, weights, paths)
    m, ld = len(ref), oracle.edits(path)
    inserted, deleted, substituted = oracle.words_of(ref, hyp, path)
    return [ld / m, oracle.pair_rate(oracle.counts(path), m, weights), ld, m, *oracle.counts(path), inserted, deleted,
            substituted]


class TestWeightedAlignment(unittest.TestCase):
    """
    This class contains unit tests for the alignment of the 'werp', 'werps' and 'summaryp' functions.
    """

    @PANDAS
    def test_matches_exhaustive_search(self):
        """
        Test that werp, werps and summaryp equal the exhaustive search on every pair of texts over three words with 0
        to 3 words a side and on a seeded sample of pairs with up to 6 words a side, for a weight set of each
        direction: the rates bit for bit, and the counts and word lists of summaryp.
        """
        failures = []
        for ref_text, hyp_text in exhaustive_pairs(3) + sample_pairs(60, 20261030):
            paths = list(oracle.alignments(ref_text.split(), hyp_text.split()))
            for weights in DIRECTION_WEIGHTS.values():
                want = expected_row(ref_text, hyp_text, weights, paths)
                got = [
                    werpy.werp(ref_text, hyp_text, *weights),
                    werpy.werps(ref_text, hyp_text, *weights),
                    summaryp_row(werpy.summaryp(ref_text, hyp_text, *weights)),
                ]
                if got != [want[1], [want[1]], want]:
                    failures.append((ref_text, hyp_text, weights, got, want))
        self.assertEqual(failures, [])

    @PANDAS
    def test_swapped_words(self):
        """
        Test two swapped words. With the weights 0.5, 0.5 and 1, one deletion and one insertion cost less than two
        substitutions. With the weights 2.5, 0.75 and 1.3, two substitutions cost less.
        """
        ref, hyp = SWAPPED
        self.assertEqual(werpy.werp(ref, hyp, 0.5, 0.5, 1), 0.2)
        self.assertEqual(werpy.werps(ref, hyp, 0.5, 0.5, 1), [0.2])
        self.assertEqual(summaryp_row(werpy.summaryp(ref, hyp, 0.5, 0.5, 1))[1:],
                         [0.2, 2, 5, 1, 1, 0, ["me"], ["me"], []])
        self.assertEqual(werpy.werp(ref, hyp, 2.5, 0.75, 1.3), 0.52)
        self.assertEqual(werpy.werps(ref, hyp, 2.5, 0.75, 1.3), [0.52])
        self.assertEqual(summaryp_row(werpy.summaryp(ref, hyp, 2.5, 0.75, 1.3))[1:],
                         [0.52, 2, 5, 0, 0, 2, [], [], [("me", "back"), ("back", "me")]])
        self.assertEqual(werpy.werp("a b", "b a", 0.5, 0.5, 1), 0.5)
        self.assertEqual(summary_counts(werpy.summaryp("a b", "b a", 0.5, 0.5, 1)), [1, 1, 0, ["a"], ["a"], []])
        self.assertEqual(werpy.werp("a b", "b a", 2.5, 0.75, 1.3), 1.3)
        self.assertEqual(summary_counts(werpy.summaryp("a b", "b a", 2.5, 0.75, 1.3)),
                         [0, 0, 2, [], [], [("a", "b"), ("b", "a")]])

    @PANDAS
    def test_most_substitutions_when_substitution_is_lighter(self):
        """
        Test that a substitution that weighs less than the mean of an insertion and a deletion gives the alignment
        with the most substitutions. For 'a a b c' and 'b c b', the alignment of summary has no substitution.
        """
        ref, hyp = LIGHTER
        self.assertEqual(summary_counts(werpy.summary(ref, hyp))[:3], [1, 2, 0])
        self.assertEqual(werpy.werp(ref, hyp, 2.5, 0.75, 1.3), 0.8375)
        self.assertEqual(werpy.werps(ref, hyp, 2.5, 0.75, 1.3), [0.8375])
        row = summaryp_row(werpy.summaryp(ref, hyp, 2.5, 0.75, 1.3))
        self.assertEqual(row[4:7], [0, 1, 2])
        self.assertEqual(row, expected_row(ref, hyp, (2.5, 0.75, 1.3)))

    @PANDAS
    def test_equal_mean_weights_keep_summary_alignment(self):
        """
        Test that summaryp gives the counts and word lists of summary when a substitution weighs exactly the mean of an
        insertion and a deletion, for pairs and for a batch of pairs with tied alignments.
        """
        refs, hyps = [pair[0] for pair in TIE_PAIRS], [pair[1] for pair in TIE_PAIRS]
        batch_summary = werpy.summary(refs, hyps)
        for weights in EQUAL_MEAN_WEIGHTS:
            batch_summaryp = werpy.summaryp(refs, hyps, *weights)
            for index, (ref, hyp) in enumerate(TIE_PAIRS):
                with self.subTest(reference=ref, hypothesis=hyp, weights=weights):
                    self.assertEqual(summary_counts(werpy.summaryp(ref, hyp, *weights)),
                                     summary_counts(werpy.summary(ref, hyp)))
                    self.assertEqual(summary_counts(batch_summaryp, index), summary_counts(batch_summary, index))

    @PANDAS
    def test_counts_add_up_to_edit_distance(self):
        """
        Test every summaryp row for every weight set: the insertions, deletions and substitutions add up to ld, wer is
        ld / m, and werp is the weighted counts divided by m.
        """
        pairs = list(TIE_PAIRS) + exhaustive_pairs(2)
        refs, hyps = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        edit_distances = [werpy.summary(ref, hyp)["ld"].iloc[0] for ref, hyp in pairs]
        failures = []
        for weights in ALL_WEIGHTS:
            frame = werpy.summaryp(refs, hyps, *weights)
            for index, (ref, hyp) in enumerate(pairs):
                wer, werp, ld, m, ins, dels, subs = summaryp_row(frame, index)[:7]
                weighted = (ins * weights[0] + dels * weights[1] + subs * weights[2]) / m
                if not (ins + dels + subs == ld == edit_distances[index] and wer == ld / m and werp == weighted):
                    failures.append((ref, hyp, weights, summaryp_row(frame, index)))
        self.assertEqual(failures, [])

    @PANDAS
    def test_not_above_summary_alignment(self):
        """
        Test that the weighted cost of the alignment of werp, werps and summaryp is never above the weighted cost of
        the alignment of summary, and that werp is never above the rate of the counts of summary.
        """
        pairs = list(TIE_PAIRS) + exhaustive_pairs(3)
        refs, hyps = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        summary_records = werpy.summary(refs, hyps).to_dict("records")
        failures = []
        for weights in ALL_WEIGHTS:
            records = werpy.summaryp(refs, hyps, *weights).to_dict("records")
            for index, (ref, hyp) in enumerate(pairs):
                today = counts_of(summary_records[index])[:3]
                chosen = counts_of(records[index])[:3]
                if (oracle.weighted_cost(chosen, weights) > oracle.weighted_cost(today, weights)
                        or werpy.werp(ref, hyp, *weights) > oracle.pair_rate(today, len(ref.split()), weights)):
                    failures.append((ref, hyp, weights, chosen, today))
        self.assertEqual(failures, [])

    @PANDAS
    def test_batches_equal_pairs(self):
        """
        Test that the rows of werps and summaryp on a batch equal their results on each pair, for lists, tuples and
        numpy arrays, and that werp on a batch is the rate of the sums of the counts and of the reference words of its
        pairs.
        """
        pairs = list(TIE_PAIRS) + sample_pairs(40, 7)
        refs, hyps = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        forms = {"list": (refs, hyps), "tuple": (tuple(refs), tuple(hyps))}
        if np is not None:
            forms["numpy str"] = (np.array(refs, dtype=str), np.array(hyps, dtype=str))
            forms["numpy object"] = (np.array(refs, dtype=object), np.array(hyps, dtype=object))
        paths = [list(oracle.alignments(ref.split(), hyp.split())) for ref, hyp in pairs]
        for weights in (*DIRECTION_WEIGHTS.values(), (0.1, 0.3, 0.7)):
            rows = []
            for (ref, hyp), pair_paths in zip(pairs, paths):
                path = oracle.choose(ref.split(), hyp.split(), weights, pair_paths)
                rows.append((*oracle.counts(path), len(ref.split())))
            singles = [werpy.werps(ref, hyp, *weights)[0] for ref, hyp in pairs]
            single_rows = [summaryp_row(werpy.summaryp(ref, hyp, *weights)) for ref, hyp in pairs]
            for form, (batch_refs, batch_hyps) in forms.items():
                with self.subTest(form=form, weights=weights):
                    self.assertEqual(werpy.werp(batch_refs, batch_hyps, *weights), oracle.batch_rate(rows, weights))
                    self.assertEqual(werpy.werps(batch_refs, batch_hyps, *weights), singles)
                    frame = werpy.summaryp(batch_refs, batch_hyps, *weights)
                    self.assertEqual([summaryp_row(frame, index) for index in range(len(pairs))], single_rows)

    @PANDAS
    def test_unweighted_results_unchanged(self):
        """
        Test that wer, wers and summary are unchanged on pairs with tied alignments: wer is the number of edits divided
        by the number of reference words, summary uses its own alignment, and werp and werps with the weights 1, 1 and
        1 equal wer and wers bit for bit.
        """
        refs, hyps = [pair[0] for pair in TIE_PAIRS], [pair[1] for pair in TIE_PAIRS]
        for ref, hyp in TIE_PAIRS:
            path = oracle.choose(ref.split(), hyp.split(), (1, 1, 1))
            with self.subTest(reference=ref, hypothesis=hyp):
                self.assertEqual(werpy.wer(ref, hyp), oracle.edits(path) / len(ref.split()))
                self.assertEqual(werpy.wers(ref, hyp), [werpy.wer(ref, hyp)])
                self.assertEqual(summary_counts(werpy.summary(ref, hyp)),
                                 [*oracle.counts(path), *oracle.words_of(ref.split(), hyp.split(), path)])
                self.assertEqual(werpy.werp(ref, hyp), werpy.wer(ref, hyp))
                self.assertEqual(werpy.werp(ref, hyp, 1, 1, 1), werpy.wer(ref, hyp))
                self.assertEqual(werpy.werps(ref, hyp, 1, 1, 1), werpy.wers(ref, hyp))
        self.assertEqual(werpy.werp(refs, hyps, 1, 1, 1), werpy.wer(refs, hyps))
        self.assertEqual(werpy.werps(refs, hyps, 1, 1, 1), werpy.wers(refs, hyps))

    def test_substitution_order_is_exact(self):
        """
        Test that substitution_order compares twice the substitution weight with the sum of the insertion and deletion
        weights exactly, for ints, floats and a mix of both, when the rounded float sum equals twice the substitution
        weight, and when both sides overflow.
        """
        largest = sys.float_info.max
        cases = [
            ((1, 1, 1), 0), ((0.5, 0.5, 1), 1), ((2.5, 0.75, 1.3), -1), ((0, 0, 0), 0), ((1, 3, 2), 0),
            ((0.5, 0.5, 0.5), 0), ((0, 0, 1), 1), ((1, 1, 0), -1), ((1, 3.0, 2), 0), ((1.0, 1, 1.5), 1),
            ((2, 2, 1.0), -1), ((0.1, 0.2, 0.15), -1), ((0.1, 0.3, 0.7), 1),
            ((1 + 2**-52, 2**-53, 0.5 + 2**-52), 1), ((1.5e308, 1.5e308, 1.5e308), 0), ((largest, 0, largest / 2), 0),
            ((largest, largest, 1), -1), ((0, 0, largest), 1), ((largest, largest, largest), 0),
            # Both sides overflow, and the weights are not equal: these rows reach the branch for a sum that overflows.
            ((1.6e308, 1.6e308, 1.7e308), 1), ((1.7e308, 1.7e308, 1.6e308), -1),
            ((2.0**1023, 1.5 * 2.0**1023, 1.25 * 2.0**1023), 0),
        ]
        # The rounded float sum equals twice the substitution weight, although the exact sum is less.
        self.assertEqual((1 + 2**-52) + 2**-53, 2 * (0.5 + 2**-52))
        for weights, order in cases:
            with self.subTest(weights=weights):
                self.assertEqual(substitution_order(*weights), order)
                self.assertEqual(oracle.direction(weights), order)

    def test_weighted_kernels(self):
        """
        Test the compiled kernels and routers of werpy._weighted with the orders 1 and -1 on an identical pair, an
        empty hypothesis and a pair of 500 words, against the dynamic programme of the reference, and that any other
        order raises ValueError. The dynamic programme is first compared with the exhaustive search.
        """
        for ref, hyp in exhaustive_pairs(3):
            for order in (1, -1):
                self.assertEqual(oracle.lex_path(ref.split(), hyp.split(), order),
                                 oracle.choose(ref.split(), hyp.split(), DIRECTION_WEIGHTS[order]))
        rng = random.Random(500)
        cases = [
            ("a b c", "a b c"),
            ("a b c", ""),
            (" ".join(rng.choice("abcd") for _ in range(500)), " ".join(rng.choice("abcd") for _ in range(500))),
        ]
        refs, hyps = [case[0] for case in cases], [case[1] for case in cases]
        for order in (1, -1):
            for ref, hyp in cases:
                path = oracle.lex_path(ref.split(), hyp.split(), order)
                m, ld = len(ref.split()), oracle.edits(path)
                full = [float(ld / m), ld, m, *oracle.counts(path), *oracle.words_of(ref.split(), hyp.split(), path)]
                with self.subTest(order=order, words=m):
                    self.assertEqual(KERNELS.calculations_weighted(ref, hyp, order), tuple(full))
                    self.assertEqual(KERNELS.metrics_weighted(ref, hyp, order), tuple(full))
                    self.assertEqual(KERNELS.calculations_fast_weighted(ref, hyp, order),
                                     tuple(float(value) for value in full[:6]))
                    self.assertEqual(KERNELS.metrics_fast_weighted(ref, hyp, order),
                                     tuple(float(value) for value in full[:6]))
            batch = KERNELS.metrics_weighted(refs, hyps, order)
            batch_fast = KERNELS.metrics_fast_weighted(refs, hyps, order)
            self.assertEqual([len(row) for row in batch], [9] * 3)
            self.assertEqual([len(row) for row in batch_fast], [6] * 3)
            for index, (ref, hyp) in enumerate(cases):
                self.assertEqual(batch[index], KERNELS.calculations_weighted(ref, hyp, order))
                self.assertEqual(batch_fast[index], KERNELS.calculations_fast_weighted(ref, hyp, order))
        for function in (KERNELS.calculations_weighted, KERNELS.metrics_weighted, KERNELS.calculations_fast_weighted,
                         KERNELS.metrics_fast_weighted):
            for order in (0, 2, -2):
                with self.subTest(function=function.__name__, order=order):
                    with self.assertRaises(ValueError) as caught:
                        function("a b", "b a", order)
                    self.assertEqual(str(caught.exception), ORDER_MESSAGE)
        for function in (KERNELS.metrics_weighted, KERNELS.metrics_fast_weighted):
            with self.assertRaises(ValueError) as caught:
                function([], [], 0)
            self.assertEqual(str(caught.exception), ORDER_MESSAGE)
            with self.assertRaises(ValueError) as caught:
                function(["a b"], ["a b", "b a"], 1)
            self.assertEqual(str(caught.exception),
                             "The Reference and Hypothesis input parameters must have the same number of elements.")

    def test_copied_helpers_agree(self):
        """
        Test that werpy._weighted's copies of the token mapping and of the length check give the results of the ones
        in werpy._metrics on the same inputs. A reference '@r <word>' and a hypothesis '@h <other word>' have one edit
        when the token mapping finds the two words the same, and two edits otherwise. The length check raises the same
        ValueError for batches of different lengths, and nothing for batches of the same length.
        """
        for first, second in itertools.product(TOKEN_WORDS, repeat=2):
            ref, hyp = f"@r {first}", f"@h {second}"
            want = 1 if first == second else 2
            with self.subTest(first=ascii(first)[:20], second=ascii(second)[:20]):
                self.assertEqual(
                    [
                        METRICS.calculations(ref, hyp)[1],
                        METRICS.calculations_fast(ref, hyp)[1],
                        METRICS.calculations_wer_only(ref, hyp)[1],
                        METRICS.metrics_wer_only([ref], [hyp])[0][1],
                        *(KERNELS.calculations_weighted(ref, hyp, order)[1] for order in (1, -1)),
                        *(KERNELS.calculations_fast_weighted(ref, hyp, order)[1] for order in (1, -1)),
                    ],
                    [want] * 8,
                )
        batches = [(["a b"], ["a b", "b a"]), (["a b", "b a"], ["a b"]), ([], ["a"]), (["a"], []), ([], []),
                   (["a b"], ["b a"])]
        for refs, hyps in batches:
            outcomes = [outcome(router, refs, hyps) for router in (METRICS.metrics, METRICS.metrics_fast,
                                                                    METRICS.metrics_wer_only)]
            outcomes += [outcome(router, refs, hyps, order) for router in (KERNELS.metrics_weighted,
                                                                           KERNELS.metrics_fast_weighted)
                         for order in (1, -1)]
            with self.subTest(reference=refs, hypothesis=hyps):
                want = (ValueError, LENGTH_MESSAGE) if len(refs) != len(hyps) else None
                self.assertEqual(outcomes, [want] * 7)

    @PANDAS
    def test_documented_examples_unchanged(self):
        """
        Test that the weighted examples of the README, the documentation and the docstrings print as documented.
        """
        ref = ["it was beautiful and sunny today"]
        hyp = ["it was a beautiful and sunny day"]
        self.assertEqual(str(werpy.werp(ref, hyp)), "0.3333333333333333")
        self.assertEqual(str(werpy.werp(ref, hyp, insertions_weight=1, deletions_weight=1, substitutions_weight=1)),
                         "0.3333333333333333")
        self.assertEqual(
            str(werpy.werp(ref, hyp, insertions_weight=0.5, deletions_weight=0.5, substitutions_weight=1)), "0.25"
        )
        self.assertEqual(str(werpy.werp(ref, hyp, 0.5, 0.5, 1)), "0.25")
        ref = ["it was beautiful and sunny today", "tomorrow may not be as nice"]
        hyp = ["it was a beautiful and sunny day", "tomorrow may not be as nice"]
        self.assertEqual(
            str(werpy.werp(ref, hyp, insertions_weight=0.5, deletions_weight=0.5, substitutions_weight=1)), "0.125"
        )
        ref = ["it blocked sight lines of central park", "her father was an alderman in the city government"]
        hyp = ["it blocked sightlines of central park", "our father was an elder man in the city government"]
        self.assertEqual(str(werpy.werps(ref, hyp)), "[0.2857142857142857, 0.3333333333333333]")
        self.assertEqual(
            str(werpy.werps(ref, hyp, insertions_weight=0.5, deletions_weight=0.5, substitutions_weight=1)),
            "[0.21428571428571427, 0.2777777777777778]",
        )
        self.assertEqual(str(werpy.werps("i love cold pizza", "i love pizza", 0.5, 0.5, 1)), "[0.125]")
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
        frame = werpy.summaryp(ref, hyp, insertions_weight=0.5, deletions_weight=0.5, substitutions_weight=1)
        rows = [summaryp_row(frame, index) for index in range(3)]
        self.assertEqual([[round(row[0], 6), round(row[1], 6)] for row in rows],
                         [[0.153846, 0.115385], [0.111111, 0.111111], [0.2, 0.15]])
        self.assertEqual([row[2:] for row in rows], [
            [2, 13, 0, 1, 1, [], ["sight"], [("lines", "sightlines")]],
            [1, 9, 0, 0, 1, [], [], [("her", "our")]],
            [2, 10, 1, 0, 1, ["balen"], [], [("ballinalee", "alley")]],
        ])

    @PANDAS
    def test_weights_compared_as_floats(self):
        """
        Test that the weights are compared as their float values, the values that the arithmetic of werp and werps
        multiplies. An int above 2**53 is compared as float(weight): 2**53 + 1 is compared as 2**53.
        """
        big = 2**53
        # As ints, 2 * (2**53 + 1) is more than 2**53 + 2**53; as floats, the two are equal.
        self.assertEqual(oracle.direction((big, big, big + 1)), 1)
        self.assertEqual(oracle.direction((float(big), float(big), float(big + 1))), 0)
        self.assertEqual(substitution_order(big, big, big + 1), 0)
        # As ints, 2 * (2**53 + 1) equals (2**53 + 2) + 2**53; as floats, it is less.
        self.assertEqual(oracle.direction((big + 2, big, big + 1)), 0)
        self.assertEqual(oracle.direction((float(big + 2), float(big), float(big + 1))), -1)
        self.assertEqual(substitution_order(big + 2, big, big + 1), -1)
        for ref, hyp in TIE_PAIRS:
            path = oracle.choose(ref.split(), hyp.split(), (float(big + 2), float(big), float(big + 1)))
            m = len(ref.split())
            with self.subTest(reference=ref, hypothesis=hyp):
                self.assertEqual(summary_counts(werpy.summaryp(ref, hyp, big, big, big + 1)),
                                 summary_counts(werpy.summary(ref, hyp)))
                self.assertEqual(summary_counts(werpy.summaryp(ref, hyp, big + 2, big, big + 1)),
                                 [*oracle.counts(path), *oracle.words_of(ref.split(), hyp.split(), path)])
                self.assertEqual(werpy.werp(ref, hyp, big + 2, big, big + 1),
                                 oracle.pair_rate(oracle.counts(path), m, (big + 2, big, big + 1)))
                self.assertEqual(werpy.werps(ref, hyp, big + 2, big, big + 1),
                                 [oracle.pair_rate(oracle.counts(path), m, (big + 2, big, big + 1))])

    def test_summaryp_werp_column_equals_werp(self):
        """
        Test that the werp column of summaryp holds, for each pair, the result of werp and of werps for that pair, bit
        for bit, also for int weights above 2**53, whose products and sums are not exact in float arithmetic. Each
        weight is used as its float value, as the oracle uses it. For some pairs, the exact rate of the int weights
        differs from that result.
        """
        big = 2**53
        pairs = list(TIE_PAIRS) + sample_pairs(40, 11)
        refs, hyps = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        exact_differs = 0
        for weights in ((big, big, big + 1), (big + 2, big, big + 1), (3, 7, 2**60 + 1), (0.1, 0.3, 0.7)):
            records = werpy.summaryp(refs, hyps, *weights, output="records")
            column = [record["werp"] for record in records]
            with self.subTest(weights=weights):
                self.assertEqual(column, werpy.werps(refs, hyps, *weights))
                self.assertEqual(column, [werpy.werp(ref, hyp, *weights) for ref, hyp in pairs])
                self.assertEqual(column, [werpy.summaryp(ref, hyp, *weights, output="records")[0]["werp"]
                                          for ref, hyp in pairs])
                self.assertEqual(column, [oracle.pair_rate(counts_of(record)[:3], record["m"], weights)
                                          for record in records])
            if all(isinstance(weight, int) for weight in weights):
                exact_differs += sum(
                    sum(count * weight for count, weight in zip(counts_of(record)[:3], weights)) / record["m"]
                    != record["werp"]
                    for record in records
                )
        self.assertGreater(exact_differs, 0)

    def test_werp_batch_rate_of_the_sums(self):
        """
        Test that werp on a batch applies the weights to the sums of the insertions, deletions and substitutions of its
        pairs and divides by the sum of their reference words, in float arithmetic, as the oracle does, for a list, a
        tuple and numpy arrays, and that a batch of one pair gives the result of that pair.
        """
        pairs = list(TIE_PAIRS) + sample_pairs(60, 13)
        refs, hyps = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        forms = [(refs, hyps), (tuple(refs), tuple(hyps))]
        if np is not None:
            forms += [(np.array(refs, dtype=str), np.array(hyps, dtype=str)),
                      (np.array(refs, dtype=object), np.array(hyps, dtype=object))]
        for weights in ((0.1, 0.3, 0.7), (2.5, 0.75, 1.3), (1, 1, 1), (2**53, 2**53, 2**53 + 1),
                        (1 + 2**-52, 2**-53, 0.5 + 2**-52)):
            records = werpy.summaryp(refs, hyps, *weights, output="records")
            rows = [(*counts_of(record)[:3], record["m"]) for record in records]
            for ref, hyp in forms:
                with self.subTest(weights=weights, form=type(ref).__name__):
                    self.assertEqual(werpy.werp(ref, hyp, *weights), oracle.batch_rate(rows, weights))
            for ref, hyp in pairs[:7]:
                with self.subTest(weights=weights, pair=(ref, hyp)):
                    self.assertEqual(werpy.werp([ref], [hyp], *weights), werpy.werp(ref, hyp, *weights))

    def test_very_large_weights(self):
        """
        Test that weights near the largest float give, in werp, werps and summaryp, the rate of the float arithmetic
        where that arithmetic gives a finite rate. Where a product or a sum of it is too large for a float, they give
        the exact rate rounded to the nearest float, or inf when the exact rate is too large for a float. No warning is
        issued.
        """
        weights = (1.5e308, 1.5e308, 1.5e308)
        # One edit gives the weight divided by the words. Two substitutions over two words overflow in float arithmetic,
        # and their exact rate is the weight. Three edits over one word give a rate too large for a float.
        pairs = [("a b c d", "a x c d"), ("a b", "x y"), ("a", "x y z"), ("a b c d e f", "a b c d e x")]
        expected = [1.5e308 / 4, 1.5e308, math.inf, 1.5e308 / 6]
        refs, hyps = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(werpy.werps(refs, hyps, *weights), expected)
            self.assertEqual([werpy.werp(ref, hyp, *weights) for ref, hyp in pairs], expected)
            self.assertEqual([record["werp"] for record in werpy.summaryp(refs, hyps, *weights, output="records")],
                             expected)
            # The batch holds 7 edits and 13 words. The sum of its float arithmetic is too large for a float, and its
            # exact rate is finite.
            self.assertEqual(werpy.werp(refs, hyps, *weights), oracle.nearest_float(7 * Fraction(1.5e308) / 13))
            self.assertEqual(werpy.werp(refs[:1], hyps[:1], *weights), 1.5e308 / 4)

    def test_overflow_cases(self):
        """
        Test each kind of input whose float arithmetic of the weighted errors overflows, with werp, werps, and summaryp
        in each form that the environment can build. Where a product or a sum is too large for a float, the result is
        the exact rate rounded to the nearest float, or inf when the exact rate is too large for a float. No warning is
        issued.
        """
        fmax, big = sys.float_info.max, 1.5e308
        # (reference, hypothesis, weights, the rate of each pair, the rate of werp)
        cases = [
            # A product overflows, and the exact rate is the weight.
            ("a b", "x y", (big, big, big), [big], big),
            # A product overflows, and the exact rate is too large for a float.
            ("a", "a x y", (big, big, big), [math.inf], math.inf),
            # The sum of one insertion and one substitution overflows, and the exact rate is too large for a float.
            ("a", "x y", (big, big, big), [math.inf], math.inf),
            # A product overflows, and the exact rate is the largest float.
            ("a b", "x y", (fmax, fmax, fmax), [fmax], fmax),
            # The sum of one substitution and one deletion overflows, and the exact rate is finite.
            ("a b c d", "a x c", (big, big, big), [big / 2], big / 2),
            # Nothing overflows.
            ("a b c d", "a x c d", (big, big, big), [big / 4], big / 4),
            # Int weights: a product overflows, and the exact rate is too large for a float.
            ("a", "x y z", (10**308, 10**308, 10**308), [math.inf], math.inf),
            # Int weights: a product overflows, and the exact rate is the float value of the weight.
            ("a b", "x y", (10**308, 10**308, 10**308), [1e308], 1e308),
            # Int weights equal to the largest float: nothing overflows.
            ("a", "x", (int(fmax), int(fmax), int(fmax)), [fmax], fmax),
            # Two pairs of one substitution over one word: the sum of the batch overflows in werp.
            (["a", "b"], ["x", "y"], (big, big, big), [big, big], big),
            # Two pairs of one substitution over four words: the sum of the batch overflows in werp, and the rate is
            # finite.
            (["a b c d", "a b c d"], ["a x c d", "a b x d"], (big, big, big), [big / 4, big / 4], big / 4),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for ref, hyp, weights, rates, rate in cases:
                with self.subTest(reference=ref, hypothesis=hyp, weights=weights):
                    self.assertEqual(werpy.werp(ref, hyp, *weights), rate)
                    self.assertEqual(werpy.werps(ref, hyp, *weights), rates)
                    columns = werp_columns(ref, hyp, weights)
                    self.assertEqual(columns, {form: rates for form in columns})

    def test_overflow_rounding(self):
        """
        Test that the exact rate is rounded once to the nearest float, also halfway between two floats, where the
        result is the float whose last binary digit is 0, and that it is inf exactly when it is at least the largest
        float plus half the distance from it to the next power of two. For random pairs and weights near the largest
        float, werp, werps and summaryp equal the oracle, which rounds with ints only, and each result of the exact
        computation is the nearest float to the exact rate.
        """
        fmax = sys.float_info.max
        # (reference, hypothesis, weights, the rate)
        edges = [
            # One insertion and one substitution over one word: the largest float plus 2**970 lies halfway between the
            # largest float and 2**1024, so the rate is too large for a float.
            ("a", "x y", (2.0**970, 1, fmax), math.inf),
            # One insertion and two substitutions over two words: the largest float plus 2**969 rounds to the largest
            # float.
            ("a b", "x y z", (2.0**970, 1, fmax), fmax),
            # One deletion and one insertion over two words: the rate lies halfway between 2**1023 - 2**970 and
            # 2**1023, and rounds to 2**1023.
            ("a b", "b x", (fmax, 2.0**970, fmax), 2.0**1023),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for ref, hyp, weights, rate in edges:
                with self.subTest(reference=ref, hypothesis=hyp, weights=weights):
                    self.assertEqual(werpy.werp(ref, hyp, *weights), rate)
                    self.assertEqual(werpy.werps(ref, hyp, *weights), [rate])
                    self.assertEqual(werp_columns(ref, hyp, weights)["records"], [rate])
            rng = random.Random(17)
            pairs = list(TIE_PAIRS) + sample_pairs(60, 19)
            refs, hyps = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            exact_results = 0
            for _ in range(12):
                weights = tuple(rng.choice((fmax, 1.5e308, 1e308, 10**308, rng.uniform(1e307, fmax), rng.random()))
                                for _ in range(3))
                records = werpy.summaryp(refs, hyps, *weights, output="records")
                rows = [(*counts_of(record)[:3], record["m"]) for record in records]
                expected = [oracle.pair_rate(row[:3], row[3], weights) for row in rows]
                with self.subTest(weights=weights):
                    self.assertEqual(werpy.werps(refs, hyps, *weights), expected)
                    self.assertEqual([record["werp"] for record in records], expected)
                    self.assertEqual([werpy.werp(ref, hyp, *weights) for ref, hyp in pairs], expected)
                    self.assertEqual(werpy.werp(refs, hyps, *weights), oracle.batch_rate(rows, weights))
                for (ins, dels, subs, m), result in zip(rows, expected):
                    float_rate = (ins * float(weights[0]) + dels * float(weights[1]) + subs * float(weights[2])) / m
                    if float_rate != math.inf:
                        continue
                    exact_results += 1
                    exact = (ins * Fraction(float(weights[0])) + dels * Fraction(float(weights[1]))
                             + subs * Fraction(float(weights[2]))) / m
                    with self.subTest(weights=weights, row=(ins, dels, subs, m)):
                        if result == math.inf:
                            self.assertGreaterEqual(exact, Fraction(fmax) + Fraction(2) ** 970)
                        else:
                            self.assertTrue(nearest(result, exact))
            self.assertGreater(exact_results, 0)


if __name__ == "__main__":
    unittest.main()
