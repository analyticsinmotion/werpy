Migrating from 3.x
==================

Some changes in werpy 4.0.0 can change how code written for werpy 3.x runs.
This page shows the main changes, with examples for werpy 3.x and werpy 4.0.0.
The changes are in the order that you find them when you upgrade.
For the full list of changes, see the `CHANGELOG <https://github.com/analyticsinmotion/werpy/blob/main/CHANGELOG.md>`_.


Installation and extras
-----------------------

Install pandas with the pandas extra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pip install werpy`` installs werpy alone.
To get the pandas DataFrame of ``summary()`` and ``summaryp()``, install the ``pandas`` extra.

*werpy 3.x*

.. code-block:: console

   $ pip install werpy

*werpy 4.0.0*

.. code-block:: console

   $ pip install "werpy[pandas]"

If pandas is not installed, ``summary()`` and ``summaryp()`` raise ``ImportError`` for the default output, ``output="pandas"``.
With a plain install of werpy, ``output="records"`` returns the results as a list of dictionaries.
For more information, see :doc:`usage/summarization`.

Install NumPy with your project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

werpy works with the NumPy that your project installs.
If your code passes NumPy arrays to werpy, install NumPy with your project.

*werpy 3.x*

.. code-block:: console

   $ pip install werpy

*werpy 4.0.0*

.. code-block:: console

   $ pip install werpy numpy


Imports and modules
-------------------

Import the functions from the package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The modules that implement the functions are private, and their names start with an underscore.
An import from a module path, such as ``from werpy.wer import wer``, raises ``ModuleNotFoundError``.

*werpy 3.x*

.. code-block:: python

   from werpy.wer import wer

   print(wer('i love cold pizza', 'i love pizza'))

*Results Output*

.. code-block:: text

   0.25

*werpy 4.0.0*

.. code-block:: python

   from werpy import wer

   print(wer('i love cold pizza', 'i love pizza'))

*Results Output*

.. code-block:: text

   0.25

Get the counts from summary()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The package exports seven functions: ``normalize()``, ``wer()``, ``wers()``, ``werp()``, ``werps()``, ``summary()`` and ``summaryp()``.
``error_handler()`` and ``metrics()`` are internal functions, and ``summary()`` returns the nine values of ``metrics()``.

*werpy 3.x*

.. code-block:: python

   import werpy

   print(werpy.metrics('i love cold pizza', 'i love pizza'))

*Results Output*

.. code-block:: text

   [0.25 1 4 0 1 0 list([]) list(['cold']) list([])]

*werpy 4.0.0*

.. code-block:: python

   import werpy

   print(werpy.summary('i love cold pizza', 'i love pizza', output="records"))

*Results Output*

.. code-block:: text

   [{'wer': 0.25, 'ld': 1, 'm': 4, 'insertions': 0, 'deletions': 1, 'substitutions': 0, 'inserted_words': [], 'deleted_words': ['cold'], 'substituted_words': []}]


Inputs
------

Inputs are checked
^^^^^^^^^^^^^^^^^^

werpy raises ``WerpyTypeError`` or ``WerpyValueError`` from ``werpy.exceptions`` for an input that is not valid.
``WerpyTypeError`` derives from ``TypeError`` and ``AttributeError``, and ``WerpyValueError`` derives from ``ValueError`` and ``ZeroDivisionError``, so existing ``except`` clauses still catch them.

.. list-table::
   :widths: 30 20
   :header-rows: 1

   * - Input
     - Exception
   * - An element that is not a ``str``
     - ``WerpyTypeError``
   * - An array that is not one-dimensional
     - ``WerpyTypeError``
   * - Empty lists in ``wer()`` and ``werp()``
     - ``WerpyValueError``
   * - A weight that is negative, NaN or infinite
     - ``WerpyValueError``
   * - A weight that is not a real number
     - ``WerpyTypeError``

*werpy 4.0.0*

.. code-block:: python

   import werpy
   from werpy.exceptions import WerpyError

   try:
       werpy.wer(['i love pizza', None], ['i love pizza', 'i love pasta'])
   except WerpyError as error:
       print(f'{type(error).__name__}: {error}')

*Results Output*

.. code-block:: text

   WerpyTypeError: Every element of reference must be a str. An element of type NoneType was found at index 1.

For more information, see :doc:`exceptions`.


Results and outputs
-------------------

wers() and werps() on a pair of strings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``wers()`` and ``werps()`` return a list for a pair of strings, with one rate in it.
To get the rate as a float, use ``wer()`` or ``werp()``, or take the first element of the list.

*werpy 3.x*

.. code-block:: python

   import werpy

   print(werpy.wers('i love cold pizza', 'i love pizza'))

*Results Output*

.. code-block:: text

   0.25

*werpy 4.0.0*

.. code-block:: python

   import werpy

   print(werpy.wers('i love cold pizza', 'i love pizza'))

*Results Output*

.. code-block:: text

   [0.25]

The weighted alignment
^^^^^^^^^^^^^^^^^^^^^^

``werp()``, ``werps()`` and ``summaryp()`` count the errors on the alignment with the lowest weighted cost among the alignments with the fewest edits.
When a substitution weighs more or less than the mean of an insertion and a deletion, the rate and the counts of ``summaryp()`` can change.

*werpy 3.x*

.. code-block:: python

   import werpy

   ref = 'please call me back tomorrow'
   hyp = 'please call back me tomorrow'
   print(werpy.werp(ref, hyp, insertions_weight=0.5, deletions_weight=0.5, substitutions_weight=1))

*Results Output*

.. code-block:: text

   0.4

*werpy 4.0.0*

.. code-block:: python

   import werpy

   ref = 'please call me back tomorrow'
   hyp = 'please call back me tomorrow'
   print(werpy.werp(ref, hyp, insertions_weight=0.5, deletions_weight=0.5, substitutions_weight=1))

*Results Output*

.. code-block:: text

   0.2

For more information, see :doc:`usage/weighted-word-error-rate`.


Normalization
-------------

Unicode punctuation in normalize()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``normalize()`` removes the Unicode punctuation as well as the ASCII punctuation, and it puts each text into Unicode Normalization Form C (NFC).
The result of an ASCII text stays the same.

*werpy 3.x*

.. code-block:: python

   import werpy

   print(werpy.normalize('It’s “done” — café'))

*Results Output*

.. code-block:: text

   it’s “done” — café

*werpy 4.0.0*

.. code-block:: python

   import werpy

   print(werpy.normalize('It’s “done” — café'))

*Results Output*

.. code-block:: text

   its done café

For the full rule, see :doc:`usage/normalization`.

Inputs of normalize()
^^^^^^^^^^^^^^^^^^^^^

``normalize()`` accepts a string, or a list, tuple or one-dimensional NumPy array of strings.
Another input, such as a pandas ``Series``, raises ``WerpyTypeError``, which derives from ``TypeError``.

*werpy 3.x*

.. code-block:: python

   import pandas as pd
   import werpy

   texts = pd.Series(["It's very popular in Antarctica.", 'The Sugar Bear character'])
   print(werpy.normalize(texts))

*Results Output*

.. code-block:: text

   ['its very popular in antarctica', 'the sugar bear character']

*werpy 4.0.0*

.. code-block:: python

   import pandas as pd
   import werpy

   texts = pd.Series(["It's very popular in Antarctica.", 'The Sugar Bear character'])
   print(werpy.normalize(texts.tolist()))

*Results Output*

.. code-block:: text

   ['its very popular in antarctica', 'the sugar bear character']
