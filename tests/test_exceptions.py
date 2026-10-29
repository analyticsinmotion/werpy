# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
test_exceptions.py

This module contains a set of unit tests for the 'werpy.exceptions' module.

The 'werpy.exceptions' module defines 'WerpyError', the base class of the exceptions that the werpy functions
raise when an input is not valid, and its two subclasses. 'WerpyTypeError' also derives from TypeError and
AttributeError, and 'WerpyValueError' also derives from ValueError and ZeroDivisionError. The tests check the
bases of each class, the except clauses that catch each class, a pickle round trip, and that the classes are not
in the namespace of the 'werpy' package.

To run the tests, execute this module as the main program.
"""

import pickle  # nosec B403
import unittest
import werpy
from werpy.exceptions import WerpyError, WerpyTypeError, WerpyValueError


def raise_and_catch(raised, clause):
    """
    Raise an instance of the class 'raised' in a try statement that has an except clause for 'clause'. Return the
    exception when that clause catches it, or None when the exception passes the clause.
    """
    try:
        try:
            raise raised("message")
        except clause as error:
            return error
    except WerpyError:
        return None
    return None


class TestExceptions(unittest.TestCase):
    """
    This class contains unit tests for the exception classes of the 'werpy.exceptions' module.
    """

    def test_werpy_error_bases(self):
        """
        Test that WerpyError derives from Exception only.
        """
        self.assertEqual(WerpyError.__bases__, (Exception,))
        self.assertEqual(WerpyError.__mro__, (WerpyError, Exception, BaseException, object))
        self.assertTrue(issubclass(WerpyError, Exception))

    def test_werpy_type_error_bases(self):
        """
        Test that WerpyTypeError derives from WerpyError, TypeError and AttributeError, in this order, and that it
        is a subclass of each of them.
        """
        self.assertEqual(WerpyTypeError.__bases__, (WerpyError, TypeError, AttributeError))
        self.assertEqual(
            WerpyTypeError.__mro__,
            (WerpyTypeError, WerpyError, TypeError, AttributeError, Exception, BaseException, object),
        )
        for base in (WerpyError, TypeError, AttributeError):
            self.assertTrue(issubclass(WerpyTypeError, base), msg=base.__name__)

    def test_werpy_value_error_bases(self):
        """
        Test that WerpyValueError derives from WerpyError, ValueError and ZeroDivisionError, in this order, and that
        it is a subclass of each of them.
        """
        self.assertEqual(WerpyValueError.__bases__, (WerpyError, ValueError, ZeroDivisionError))
        self.assertEqual(
            WerpyValueError.__mro__,
            (
                WerpyValueError,
                WerpyError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                Exception,
                BaseException,
                object,
            ),
        )
        for base in (WerpyError, ValueError, ZeroDivisionError):
            self.assertTrue(issubclass(WerpyValueError, base), msg=base.__name__)

    def test_except_werpy_error(self):
        """
        Test that an except clause for WerpyError catches WerpyTypeError and WerpyValueError.
        """
        self.assertIs(type(raise_and_catch(WerpyTypeError, WerpyError)), WerpyTypeError)
        self.assertIs(type(raise_and_catch(WerpyValueError, WerpyError)), WerpyValueError)

    def test_except_type_error(self):
        """
        Test that an except clause for TypeError catches WerpyTypeError and does not catch WerpyValueError.
        """
        self.assertIs(type(raise_and_catch(WerpyTypeError, TypeError)), WerpyTypeError)
        self.assertIsNone(raise_and_catch(WerpyValueError, TypeError))

    def test_except_attribute_error(self):
        """
        Test that an except clause for AttributeError catches WerpyTypeError and does not catch WerpyValueError.
        """
        self.assertIs(type(raise_and_catch(WerpyTypeError, AttributeError)), WerpyTypeError)
        self.assertIsNone(raise_and_catch(WerpyValueError, AttributeError))

    def test_except_value_error(self):
        """
        Test that an except clause for ValueError catches WerpyValueError and does not catch WerpyTypeError.
        """
        self.assertIs(type(raise_and_catch(WerpyValueError, ValueError)), WerpyValueError)
        self.assertIsNone(raise_and_catch(WerpyTypeError, ValueError))

    def test_except_zero_division_error(self):
        """
        Test that an except clause for ZeroDivisionError catches WerpyValueError and does not catch
        WerpyTypeError.
        """
        self.assertIs(type(raise_and_catch(WerpyValueError, ZeroDivisionError)), WerpyValueError)
        self.assertIsNone(raise_and_catch(WerpyTypeError, ZeroDivisionError))

    def test_pickle_round_trip(self):
        """
        Test that a pickle round trip of each class keeps the class and the message.
        """
        message = "Invalid input: reference must not be blank."
        for cls in (WerpyError, WerpyTypeError, WerpyValueError):
            restored = pickle.loads(pickle.dumps(cls(message)))  # nosec B301
            self.assertIs(type(restored), cls)
            self.assertEqual(str(restored), message)
            self.assertEqual(restored.args, (message,))

    def test_classes_are_not_in_the_package_namespace(self):
        """
        Test that werpy.exceptions lists the three classes in __all__, and that the classes are neither in
        werpy.__all__ nor attributes of the werpy package.
        """
        self.assertEqual(werpy.exceptions.__all__, ["WerpyError", "WerpyTypeError", "WerpyValueError"])
        for name in werpy.exceptions.__all__:
            self.assertNotIn(name, werpy.__all__)
            self.assertFalse(hasattr(werpy, name), msg=name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
