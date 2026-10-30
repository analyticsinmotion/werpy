# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
The exceptions module defines the exceptions that the werpy functions raise when an input is not valid.

This module defines the following exceptions:
    - WerpyError: the base class of every werpy input error.
    - WerpyTypeError: raised when an input has the wrong type or kind.
    - WerpyValueError: raised when an input has a value that is not valid.
"""

__all__ = [
    "WerpyError",
    "WerpyTypeError",
    "WerpyValueError",
]


class WerpyError(Exception):
    """
    The base class of every exception that werpy raises when an input is not valid.

    werpy does not raise WerpyError directly. It raises one of its subclasses, WerpyTypeError or WerpyValueError.
    An except clause for WerpyError catches both.
    """


class WerpyTypeError(WerpyError, TypeError, AttributeError):
    """
    Raised when an input has the wrong type or kind: when a reference, hypothesis or text is not a string, a list, a
    tuple or a numpy array, when a numpy array does not have exactly one dimension, when one of the reference and the
    hypothesis is a string and the other is a list, a tuple or a numpy array, when an element of a list, tuple or
    numpy array is not a string, or when a weight of werp(), werps() or summaryp() is not a real number.

    WerpyTypeError also derives from AttributeError, so that code written for werpy 3.x that catches
    AttributeError continues to catch it.
    """


class WerpyValueError(WerpyError, ValueError, ZeroDivisionError):
    """
    Raised when an input has a value that is not valid: when a reference is blank, either as a single string or as
    an element of a list, tuple or numpy array, when the reference and hypothesis lists, tuples or arrays do not
    contain the same number of elements, when wer() or werp() gets empty lists, tuples or arrays, or when a weight of
    werp(), werps() or summaryp() is negative, NaN, infinite or too large for a float.

    WerpyValueError also derives from ZeroDivisionError, so that code written for werpy 3.x that catches
    ZeroDivisionError continues to catch it.
    """
