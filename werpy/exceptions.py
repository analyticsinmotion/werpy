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
    Raised when the reference or the hypothesis has the wrong type or kind: when it is not a string, a list or a
    numpy array, or when one of them is a string and the other is a list or a numpy array.

    WerpyTypeError also derives from AttributeError, so that code written for werpy 3.x that catches
    AttributeError continues to catch it.
    """


class WerpyValueError(WerpyError, ValueError, ZeroDivisionError):
    """
    Raised when the reference or the hypothesis has a value that is not valid: when a reference is blank, either as
    a single string or as an element of a list or numpy array, or when the reference and hypothesis lists or arrays
    do not contain the same number of elements.

    WerpyValueError also derives from ZeroDivisionError, so that code written for werpy 3.x that catches
    ZeroDivisionError continues to catch it.
    """
