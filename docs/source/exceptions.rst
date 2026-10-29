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
     - Raised when the reference or the hypothesis is not a string, a list or a NumPy array, or when one of them is a string and the other is a list or a NumPy array. It derives from WerpyError, TypeError and AttributeError.
   * - WerpyValueError
     - Raised when a reference is blank, either as a single string or as an element of a list or NumPy array, or when the reference and hypothesis lists or arrays do not contain the same number of elements. It derives from WerpyError, ValueError and ZeroDivisionError.

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
