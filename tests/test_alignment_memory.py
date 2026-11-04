# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_alignment_memory.py

This module contains a set of unit tests for the memory of the alignments on which the 'summary', 'summaryp', 'werp'
and 'werps' functions of the 'werpy' package count the errors.

The compiled kernels keep the whole table of the dynamic programme only for a short pair. For a longer pair they keep
the move that the backtrace takes at each cell, and when these moves do not fit in a fixed size, they split the pair
into pieces at cells of the backtrace and align each piece in the same way. The function 'table_budget' of
'werpy._metrics' and 'werpy._weighted' sets the two sizes, so that the tests reach the moves and the pieces with short
texts.

The tests check that the results do not depend on the two sizes, that they equal the exhaustive search of
'alignment_reference.py' when the kernels keep only the moves, that the memory of a long pair stays far below the size
of its whole table, and that the copies of the moves and the pieces in the two modules give the same results.

To run the tests, execute this module as the main program.
"""

import contextlib
import importlib
import importlib.util
import itertools
import random
import tracemalloc
import unittest
import alignment_reference as oracle
import werpy


METRICS = importlib.import_module("werpy._metrics")
KERNELS = importlib.import_module("werpy._weighted")
# The tests that read the DataFrames of summary and summaryp need pandas, an optional dependency of werpy.
PANDAS = unittest.skipIf(importlib.util.find_spec("pandas") is None, "pandas is not installed")
# Sizes in bytes of the whole table and of the moves: pieces down to one reference word; the moves of pieces of 1 and 5
# reference words with up to 32 hypothesis words; the moves of the whole pair in one pass; and small whole tables with
# small moves, so that the pairs of one batch take different ways.
SIZES = ((0, 0), (0, 8), (0, 40), (0, 2**30), (64, 8), (200, 16))
PIECES, MOVES = (0, 0), (0, 2**30)
# One weight set of each direction: a substitution heavier than the mean, lighter than the mean, and exactly the mean.
DIRECTION_WEIGHTS = {1: (0.5, 0.5, 1), -1: (2.5, 0.75, 1.3), 0: (1, 3, 2)}
SUMMARY_COLUMNS = ("wer", "ld", "m", "insertions", "deletions", "substitutions", "inserted_words", "deleted_words",
                   "substituted_words")
SUMMARYP_COLUMNS = ("wer", "werp", *SUMMARY_COLUMNS[1:])


def texts(max_words):
    """Return every text of 0 to max_words words over the words a, b and c."""
    return [" ".join(words) for count in range(max_words + 1) for words in itertools.product("abc", repeat=count)]


def exhaustive_pairs(max_words, blank=False):
    """Return every pair of texts of 0 to max_words words over a, b and c, with a blank reference only when blank."""
    return [(ref, hyp) for ref in texts(max_words) for hyp in texts(max_words) if ref or blank]


def sample_pairs(count, seed, max_words):
    """
    Return count pairs of random texts over 2 to 5 distinct words, with 1 to max_words reference words and 0 to
    max_words hypothesis words.
    """
    rng = random.Random(seed)
    pairs = []
    for _ in range(count):
        words = "abcde"[: rng.randint(2, 5)]
        ref = " ".join(rng.choice(words) for _ in range(rng.randint(1, max_words)))
        hyp = " ".join(rng.choice(words) for _ in range(rng.randint(0, max_words)))
        pairs.append((ref, hyp))
    return pairs


def edited_pair(count, seed):
    """
    Return a text of count words over the 26 one-letter words a to z and a copy of it in which about one word in eight
    is substituted, deleted or followed by an inserted word, and about one pair of neighbours in twenty is swapped.
    """
    rng = random.Random(seed)
    vocabulary = "abcdefghijklmnopqrstuvwxyz"
    ref = [rng.choice(vocabulary) for _ in range(count)]
    hyp = []
    for word in ref:
        roll = rng.random()
        if roll < 0.04:
            hyp.append(rng.choice(vocabulary))
        elif roll < 0.08:
            hyp.extend([word, rng.choice(vocabulary)])
        elif roll >= 0.12:
            hyp.append(word)
    for index in range(len(hyp) - 1):
        if rng.random() < 0.05:
            hyp[index], hyp[index + 1] = hyp[index + 1], hyp[index]
    return " ".join(ref), " ".join(hyp)


@contextlib.contextmanager
def budget(table, codes):
    """Set the two sizes of both modules for the body of a with statement, and restore them after it."""
    previous = METRICS.table_budget(table, codes), KERNELS.table_budget(table, codes)
    try:
        yield
    finally:
        METRICS.table_budget(*previous[0])
        KERNELS.table_budget(*previous[1])


def kernel_rows(ref, hyp):
    """Return the rows of calculations, calculations_fast and calculations_weighted with the orders 1 and -1."""
    return [
        plain(METRICS.calculations(ref, hyp)),
        plain(METRICS.calculations_fast(ref, hyp)),
        plain(KERNELS.calculations_weighted(ref, hyp, 1)),
        plain(KERNELS.calculations_weighted(ref, hyp, -1)),
    ]


def batch_results(pairs):
    """
    Return the rows of the routers of werpy._metrics and werpy._weighted for the pairs as one batch, and the rates of
    werp and werps with equal weights for the pairs with a non-blank reference as one batch.
    """
    refs, hyps = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    nonblank = [pair for pair in pairs if pair[0]]
    rated_refs, rated_hyps = [pair[0] for pair in nonblank], [pair[1] for pair in nonblank]
    return [
        plain(METRICS.metrics(refs, hyps)),
        plain(METRICS.metrics_fast(refs, hyps)),
        plain(KERNELS.metrics_weighted(refs, hyps, 1)),
        plain(KERNELS.metrics_weighted(refs, hyps, -1)),
        werpy.werp(rated_refs, rated_hyps),
        werpy.werps(rated_refs, rated_hyps),
    ]


def pair_results(pairs):
    """
    Return the kernel rows of every pair, and the rates of werp and werps with equal weights for a pair with a
    non-blank reference.
    """
    results = []
    for ref, hyp in pairs:
        rates = [werpy.werp(ref, hyp), werpy.werps(ref, hyp)] if ref else []
        results.append([kernel_rows(ref, hyp), rates])
    return results


def expected_row(ref, hyp, weights, paths):
    """
    Return the summaryp row that the exhaustive search gives for a pair of word lists and weights: wer, werp, ld, m, the
    counts and the word lists. The werp column is the rate of 'werp' on a pair.
    """
    path = oracle.choose(ref, hyp, weights, paths)
    m, ld = len(ref), oracle.edits(path)
    return [ld / m, oracle.pair_rate(oracle.counts(path), m, weights), ld, m, *oracle.counts(path),
            *oracle.words_of(ref, hyp, path)]


def frame_rows(frame, columns):
    """Return the rows of a summary or summaryp frame as lists of Python values, in the order of the columns given."""
    kinds = {column: list if column.endswith("_words") else float if column.startswith("wer") else int
             for column in columns}
    return [[kinds[column](record[column]) for column in columns] for record in frame.to_dict("records")]


def plain(value):
    """Return a row of a kernel as a list, a list of rows as a list of lists, and any other value unchanged."""
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def traced_peak(call, *args):
    """Return the result of call(*args) and the peak of the memory that tracemalloc traces during the call."""
    tracemalloc.start()
    try:
        result = call(*args)
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


class TestAlignmentMemory(unittest.TestCase):
    """
    This class contains unit tests for the memory of the alignments of the 'summary', 'summaryp', 'werp' and 'werps'
    functions.
    """

    def test_moves_and_pieces_equal_whole_table(self):
        """
        Test that the kernels and routers of werpy._metrics and werpy._weighted, and werp and werps with equal weights,
        give at every pair of sizes the results that they give at the default sizes, single and in batches: on every
        pair of texts of 0 to 2 words over three words, and on 24 seeded pairs of up to 40 words, so that pieces have
        four or more reference words and rows of 32 or more moves. table_budget returns the sizes that it replaces and
        leaves a size of -1 unchanged.
        """
        pairs = exhaustive_pairs(2, blank=True) + sample_pairs(24, 20261101, 40)
        defaults = METRICS.table_budget(), KERNELS.table_budget()
        want_pairs, want_batches = pair_results(pairs), batch_results(pairs)
        for table, codes in SIZES:
            with budget(table, codes):
                current = METRICS.table_budget(-1, -1), KERNELS.table_budget(-1, -1)
                got_pairs, got_batches = pair_results(pairs), batch_results(pairs)
            failures = [pair for pair, got, want in zip(pairs, got_pairs, want_pairs) if got != want]
            with self.subTest(table=table, codes=codes):
                self.assertEqual(current, ((table, codes), (table, codes)))
                self.assertEqual(failures, [])
                self.assertEqual(got_batches, want_batches)
        self.assertEqual((METRICS.table_budget(), KERNELS.table_budget()), defaults)

    @PANDAS
    def test_moves_and_pieces_match_exhaustive_search(self):
        """
        Test that summary, summaryp with a weight set of each direction, werp and werps with equal weights, and the
        counts of calculations_fast equal the exhaustive search when every pair is aligned in pieces down to one
        reference word and when the moves of the whole pair are kept: the counts, the word lists in order and the
        rates, bit for bit, on every pair of texts over three words with 0 to 3 words a side and a non-blank
        reference, and on a seeded sample of pairs with up to 6 words a side. summary, summaryp and werps take the
        pairs as one batch, whose rows are aligned one by one, and werp and calculations_fast take each pair.
        """
        pairs = exhaustive_pairs(3) + sample_pairs(60, 20261030, 6)
        refs, hyps = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        wants = []
        for ref_text, hyp_text in pairs:
            ref, hyp = ref_text.split(), hyp_text.split()
            paths = list(oracle.alignments(ref, hyp))
            unit = expected_row(ref, hyp, (1, 1, 1), paths)
            weighted = [expected_row(ref, hyp, weights, paths) for weights in DIRECTION_WEIGHTS.values()]
            wants.append([unit[:1] + unit[2:], *weighted, unit[1], weighted[2][1], unit[1],
                          [float(value) for value in unit[:1] + unit[2:7]]])
        for table, codes in (PIECES, MOVES):
            with budget(table, codes):
                columns = [
                    frame_rows(werpy.summary(refs, hyps), SUMMARY_COLUMNS),
                    *(frame_rows(werpy.summaryp(refs, hyps, *weights), SUMMARYP_COLUMNS)
                      for weights in DIRECTION_WEIGHTS.values()),
                    werpy.werps(refs, hyps),
                    werpy.werps(refs, hyps, *DIRECTION_WEIGHTS[0]),
                    [werpy.werp(ref, hyp) for ref, hyp in pairs],
                    [plain(METRICS.calculations_fast(ref, hyp)) for ref, hyp in pairs],
                ]
            failures = [(pair, list(got), want) for pair, got, want in zip(pairs, zip(*columns), wants)
                        if list(got) != want]
            with self.subTest(table=table, codes=codes):
                self.assertEqual(len(columns[0]), len(pairs))
                self.assertEqual(failures, [])

    def test_long_texts_use_bounded_memory(self):
        """
        Test that two texts of 200 words over three words, so that most cells tie, are aligned in pieces when the moves
        are limited so that the pair needs one split. The peak of the memory that tracemalloc traces during
        calculations, calculations_fast, and werp and werps with equal weights stays below a quarter of the memory of
        the whole table, and during calculations_weighted with the orders 1 and -1 below an eighth of the memory of the
        whole key matrix. The words have one letter, so that the words of the texts take no memory of their own.
        Each result, and the result for a text of 200 words with an edited copy of it, equals the result with the
        whole table.
        """
        rng = random.Random(200)
        pairs = [
            (" ".join(rng.choice("abc") for _ in range(200)), " ".join(rng.choice("abc") for _ in range(200))),
            edited_pair(200, 201),
        ]
        calls = {
            "calculations": METRICS.calculations,
            "calculations_fast": METRICS.calculations_fast,
            "calculations_weighted 1": lambda ref, hyp: KERNELS.calculations_weighted(ref, hyp, 1),
            "calculations_weighted -1": lambda ref, hyp: KERNELS.calculations_weighted(ref, hyp, -1),
            "werp": werpy.werp,
            "werps": werpy.werps,
        }
        for index, (ref, hyp) in enumerate(pairs):
            cells = (len(ref.split()) + 1) * (len(hyp.split()) + 1)
            for name, call in calls.items():
                with budget(2**30, -1):
                    whole = call(ref, hyp)
                with budget(0, 2**13):
                    # tracemalloc slows the line-trace build of the coverage job, so it watches the first pair only
                    got, peak = traced_peak(call, ref, hyp) if index == 0 else (call(ref, hyp), None)
                with self.subTest(pair=index, function=name):
                    self.assertEqual(plain(got), plain(whole))
                    if peak is not None:
                        self.assertLess(peak, cells)

    def test_copied_pieces_agree(self):
        """
        Test that the copies of the moves and the pieces in werpy._metrics and werpy._weighted give the same results.
        On every pair of texts of 0 to 4 words over three words whose alignments with the fewest edits all have the same
        number of substitutions, calculations and calculations_weighted with the orders 1 and -1 return the same row
        when every pair is aligned in pieces down to one reference word and when the moves of the whole pair are kept.
        """
        pairs = []
        for ref, hyp in exhaustive_pairs(4, blank=True):
            fewest = oracle.counts(oracle.lex_path(ref.split(), hyp.split(), 1))[2]
            most = oracle.counts(oracle.lex_path(ref.split(), hyp.split(), -1))[2]
            if fewest == most:
                pairs.append((ref, hyp))
        for table, codes in (PIECES, MOVES):
            with budget(table, codes):
                failures = [
                    (ref, hyp) for ref, hyp in pairs
                    if not (METRICS.calculations(ref, hyp)
                            == KERNELS.calculations_weighted(ref, hyp, 1)
                            == KERNELS.calculations_weighted(ref, hyp, -1))
                ]
            with self.subTest(table=table, codes=codes):
                self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
