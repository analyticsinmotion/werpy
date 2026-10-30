# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
alignment_reference.py

This module is a pure-Python reference for the word alignments of the 'werpy' package. The tests in
'test_weighted_alignment.py' compare 'werp', 'werps' and 'summaryp' with it. It imports nothing from 'werpy'. The rates
use numpy as werpy does, so that they can be compared bit for bit.

Every function takes the reference and the hypothesis as lists of words, as str.split() gives them. An alignment is a
path through the (m + 1) x (n + 1) grid from (m, n) back to (0, 0), where m and n are the numbers of reference and
hypothesis words. Each step is one move:

    M  match         reference word i and hypothesis word j are equal: (i, j) -> (i - 1, j - 1)
    S  substitution  they differ: (i, j) -> (i - 1, j - 1)
    I  insertion     hypothesis word j has no reference word: (i, j) -> (i, j - 1)
    D  deletion      reference word i has no hypothesis word: (i, j) -> (i - 1, j)

A path is a tuple of moves read from the end. Its number of edits is I + D + S. The backtrace order compares two paths
at the first step, from the end, where they differ: M before S before I before D.

The rule of 'werp', 'werps' and 'summaryp': among the alignments with the fewest edits, the alignments with the lowest
weighted cost wi * I + wd * D + ws * S; among those, the first in the backtrace order. Weighted costs are compared
exactly, as fractions of the weights given.
"""

from fractions import Fraction

import numpy as np

M, S, I, D = "M", "S", "I", "D"
ORDER = {M: 0, S: 1, I: 2, D: 3}


def alignments(ref, hyp):
    """Yield every alignment of ref and hyp as a tuple of moves read from the end. Nothing is pruned."""

    def walk(i, j):
        if i == 0 and j == 0:
            yield ()
            return
        if i and j:
            move = M if ref[i - 1] == hyp[j - 1] else S
            for rest in walk(i - 1, j - 1):
                yield (move,) + rest
        if j:
            for rest in walk(i, j - 1):
                yield (I,) + rest
        if i:
            for rest in walk(i - 1, j):
                yield (D,) + rest

    yield from walk(len(ref), len(hyp))


def counts(path):
    """Return the (insertions, deletions, substitutions) of a path."""
    return path.count(I), path.count(D), path.count(S)


def edits(path):
    """Return the number of edits of a path."""
    return len(path) - path.count(M)


def order_key(path):
    """Return the key of a path in the backtrace order: the smaller key comes first."""
    return tuple(ORDER[move] for move in path)


def weighted_cost(cnt, weights):
    """Return the exact weighted cost of (insertions, deletions, substitutions)."""
    (ins, dels, subs), (wi, wd, ws) = cnt, weights
    return Fraction(wi) * ins + Fraction(wd) * dels + Fraction(ws) * subs


def choose(ref, hyp, weights, paths=None):
    """
    Return the path of the rule by an exhaustive search over every alignment: the fewest edits, then the lowest
    weighted cost, then the first in the backtrace order. paths, when given, must be list(alignments(ref, hyp)).
    """
    paths = list(alignments(ref, hyp)) if paths is None else paths
    fewest = min(edits(path) for path in paths)
    paths = [path for path in paths if edits(path) == fewest]
    costs = {cnt: weighted_cost(cnt, weights) for cnt in {counts(path) for path in paths}}
    lowest = min(costs.values())
    paths = [path for path in paths if costs[counts(path)] == lowest]
    return min(paths, key=order_key)


def direction(weights):
    """
    Return the sign of 2 * ws - (wi + wd), exactly: 1 when a substitution weighs more than the mean of an insertion and
    a deletion, -1 when it weighs less, 0 when it weighs exactly the mean.
    """
    wi, wd, ws = (Fraction(weight) for weight in weights)
    diff = 2 * ws - wi - wd
    return (diff > 0) - (diff < 0)


def lex_path(ref, hyp, mode):
    """
    Return the path of the rule for a direction mode of 1 or -1 by a dynamic programme and a backtrace.

    The key of a cell is big * edits + mode * substitutions, with big larger than any number of substitutions, so that
    the keys order the alignments by edits first and then by substitutions: the fewest for mode 1 and the most for
    mode -1. The backtrace keeps the order M or S, then I, then D, and takes the first move that keeps the key optimal.
    """
    if mode not in (1, -1):
        raise ValueError(f"mode must be 1 or -1, not {mode}")
    m, n = len(ref), len(hyp)
    big = min(m, n) + 1
    sub_step = big + mode
    key = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        key[i][0] = big * i
    for j in range(n + 1):
        key[0][j] = big * j
    for i in range(1, m + 1):
        word, row, prev = ref[i - 1], key[i], key[i - 1]
        for j in range(1, n + 1):
            step = 0 if word == hyp[j - 1] else sub_step
            row[j] = min(prev[j] + big, row[j - 1] + big, prev[j - 1] + step)
    i, j = m, n
    path = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = ref[i - 1] == hyp[j - 1]
            if key[i][j] == key[i - 1][j - 1] + (0 if same else sub_step):
                path.append(M if same else S)
                i, j = i - 1, j - 1
                continue
        if j > 0 and key[i][j] == key[i][j - 1] + big:
            path.append(I)
            j -= 1
        elif i > 0 and key[i][j] == key[i - 1][j] + big:
            path.append(D)
            i -= 1
        else:
            raise AssertionError("no move keeps the key optimal")
    return tuple(path)


def words_of(ref, hyp, path):
    """Return the inserted, deleted and substituted words of a path, in text order, as 'summaryp' lists them."""
    i, j = len(ref), len(hyp)
    inserted, deleted, substituted = [], [], []
    for move in path:
        if move == M:
            i, j = i - 1, j - 1
        elif move == S:
            substituted.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif move == I:
            inserted.append(hyp[j - 1])
            j -= 1
        else:
            deleted.append(ref[i - 1])
            i -= 1
    inserted.reverse()
    deleted.reverse()
    substituted.reverse()
    return inserted, deleted, substituted


def pair_rate(cnt, m, weights):
    """
    Return the weighted rate of one pair with the arithmetic of 'werp' on a pair of strings: float64 products of the
    insertions, deletions and substitutions, summed in that order, then divided by m; 0.0 when m is 0.
    """
    (ins, dels, subs), (wi, wd, ws) = cnt, weights
    err = np.float64(ins) * wi + np.float64(dels) * wd + np.float64(subs) * ws
    return float(err / np.float64(m)) if m else 0.0


def batch_rate(rows, weights):
    """Return the rate of 'werp' on a batch of rows (insertions, deletions, substitutions, m), with its arithmetic."""
    wi, wd, ws = weights
    table = np.array(rows, dtype=np.float64).reshape(-1, 4)
    err = table[:, 0] * wi + table[:, 1] * wd + table[:, 2] * ws
    den = np.sum(table[:, 3])
    return float(np.sum(err) / den) if den else 0.0
