Exceptions
==========

The werpy functions raise an exception when an input is not valid. The exception classes are defined in the ``werpy.exceptions`` module.

Exceptions available in werpy
-----------------------------

The following table provides an overview of the exceptions that werpy raises:

.. list-table::
   :widths: 18 32
   :header-rows: 1

   * - Exception
     - Description
   * - WerpyError
     - The base class of every exception that werpy raises when an input is not valid. werpy does not raise WerpyError directly, and an except clause for WerpyError catches both of the exceptions below.
   * - WerpyTypeError
     - Raised when a reference, hypothesis or text is not a string, a list, a tuple or a NumPy array, when a NumPy array does not have exactly one dimension, when one of the reference and the hypothesis is a string and the other is a list, a tuple or a NumPy array, when an element of a list, tuple or NumPy array is not a string, or when a weight is not a real number. It derives from WerpyError, TypeError and AttributeError.
   * - WerpyValueError
     - Raised when a reference is blank, either as a single string or as an element of a list, tuple or NumPy array, when the reference and hypothesis lists, tuples or arrays do not contain the same number of elements, when wer() or werp() gets empty lists, tuples or arrays, or when a weight is negative, NaN, infinite or too large for a float. It derives from WerpyError, ValueError and ZeroDivisionError.

Code written for werpy 3.x that catches AttributeError or ZeroDivisionError continues to catch these exceptions.

Handle an invalid input
-----------------------

*Python Code*

.. code-block:: python

   import werpy
   from werpy.exceptions import WerpyError

   try:
       wer = werpy.wer('', 'i love pizza')
   except WerpyError as error:
       print(f'{type(error).__name__}: {error}')

*Results Output*

.. code-block:: python

   WerpyValueError: Invalid input: reference must not be blank, and reference and hypothesis cannot both be empty.
