# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
This module provides the forms of the results of the summary and summaryp functions. With output="pandas", the
default, the results are a pandas DataFrame. With output="polars", they are a polars DataFrame. With output="records",
they are a list of dictionaries, one for each pair of texts. This form works with a plain install of werpy.

pandas and polars are optional dependencies of werpy, which the pandas and polars extras install: pip install
"werpy[pandas]" and pip install "werpy[polars]". This module imports pandas when werpy is imported, if it can be
imported. It imports polars when output="polars" is requested.

This module defines the following functions:
    - check_output(function, output)
    - dataframe(rows, columns)
    - polars_dataframe(rows, columns)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

from .exceptions import WerpyTypeError, WerpyValueError

try:
    import pandas as pd
except ImportError as error:
    # pandas is optional. The error is kept, so that the exception for output="pandas" shows its cause.
    pd = None  # type: ignore[assignment]
    PANDAS_ERROR: ImportError | None = error
else:
    PANDAS_ERROR = None

if TYPE_CHECKING:
    import polars as pl

    # The DataFrame types of the results of summary() and summaryp().
    PandasDataFrame: TypeAlias = pd.DataFrame
    PolarsDataFrame: TypeAlias = pl.DataFrame
else:
    # At run time, the aliases name typing.Any for the DataFrame of a library that werpy has not imported, so that
    # typing.get_type_hints() resolves the annotations of summary() and summaryp() in every environment.
    PandasDataFrame: TypeAlias = Any if pd is None else pd.DataFrame
    PolarsDataFrame: TypeAlias = Any


# The columns of the results of summary() and summaryp(), in order. They are the keys of SummaryRecord and
# SummarypRecord.
SUMMARY_COLUMNS = [
    "wer",
    "ld",
    "m",
    "insertions",
    "deletions",
    "substitutions",
    "inserted_words",
    "deleted_words",
    "substituted_words",
]
SUMMARYP_COLUMNS = [
    "wer",
    "werp",
    "ld",
    "m",
    "insertions",
    "deletions",
    "substitutions",
    "inserted_words",
    "deleted_words",
    "substituted_words",
]


class SummaryRecord(TypedDict):
    """
    The results of summary() for one pair of texts with output="records". The keys are the columns of the DataFrame
    of summary(), in the same order.
    """

    wer: float
    ld: int
    m: int
    insertions: int
    deletions: int
    substitutions: int
    inserted_words: list[str]
    deleted_words: list[str]
    substituted_words: list[tuple[str, str]]


class SummarypRecord(TypedDict):
    """
    The results of summaryp() for one pair of texts with output="records". The keys are the columns of the DataFrame
    of summaryp(), in the same order.
    """

    wer: float
    werp: float
    ld: int
    m: int
    insertions: int
    deletions: int
    substitutions: int
    inserted_words: list[str]
    deleted_words: list[str]
    substituted_words: list[tuple[str, str]]


def check_output(function: str, output: object) -> str:
    """
    Check the output argument of summary() or summaryp(), and return it.

    WerpyTypeError is raised for an output that is not a string, and WerpyValueError for a string that is not
    "pandas", "polars" or "records". ImportError is raised for "pandas" when pandas could not be imported, and for
    "polars" when polars could not be imported. Its message names the function, the extra that installs the library
    and the other values of output, and its cause is the error of the import of the library.
    """
    if not isinstance(output, str):
        raise WerpyTypeError(f'output must be "pandas", "polars" or "records", not {type(output).__name__}.')
    if output == "pandas":
        if pd is None:
            raise ImportError(
                f"{function}() returns a pandas DataFrame, and pandas could not be imported. Install pandas, for "
                f'example with pip install "werpy[pandas]", or call {function}() with output="polars" to get a polars '
                'DataFrame or with output="records" to get a list of dictionaries.',
                name="pandas",
            ) from PANDAS_ERROR
    elif output == "polars":
        try:
            import polars  # pylint: disable=import-outside-toplevel,unused-import
        except ImportError as error:
            raise ImportError(
                f'{function}() with output="polars" returns a polars DataFrame, and polars could not be imported. '
                f'Install polars, for example with pip install "werpy[polars]", or call {function}() with '
                'output="records" to get a list of dictionaries.',
                name="polars",
            ) from error
    elif output != "records":
        raise WerpyValueError(f'output must be "pandas", "polars" or "records", not "{output}".')
    return output


def dataframe(rows: Sequence[Sequence[object]], columns: list[str]) -> pd.DataFrame:
    """
    Return the pandas DataFrame of the rows of summary() or summaryp(), with the given columns. check_output() is
    called first, so pandas can be imported.
    """
    return pd.DataFrame(rows, columns=columns)


def polars_dataframe(rows: Sequence[Sequence[Any]], columns: list[str]) -> pl.DataFrame:
    """
    Return the polars DataFrame of the rows of summary() or summaryp(), with the given columns, of which the last is
    substituted_words. The rates are Float64, the counts are Int64, the inserted and deleted words are lists of String,
    and each substitution is a Struct with the fields reference and hypothesis. check_output() is called first, so
    polars can be imported.
    """
    import polars  # pylint: disable=import-outside-toplevel

    words = polars.List(polars.String())
    types: dict[str, polars.DataType] = {
        "wer": polars.Float64(),
        "werp": polars.Float64(),
        "inserted_words": words,
        "deleted_words": words,
        "substituted_words": polars.List(polars.Struct({"reference": polars.String(), "hypothesis": polars.String()})),
    }
    # Each substitution is given to polars as a dictionary with the names of the fields of the Struct. From a tuple in
    # a row, some versions of polars give the fields as null, and others raise an error.
    return polars.DataFrame(
        [[*row[:-1], [{"reference": ref, "hypothesis": hyp} for ref, hyp in row[-1]]] for row in rows],
        schema={name: types.get(name, polars.Int64()) for name in columns},
        orient="row",
    )
