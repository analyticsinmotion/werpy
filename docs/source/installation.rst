Installation
============

pip (recommended)
-----------------

You can install werpy with Python's pip package manager:

.. code-block:: console

   $ pip install werpy

This will install the latest stable release of werpy.

summary() and summaryp() return their results as a pandas DataFrame, a polars DataFrame or a list of dictionaries. To install pandas or polars with werpy, use its extra:

.. code-block:: console

   $ pip install "werpy[pandas]"
   $ pip install "werpy[polars]"

With a plain pip install werpy, output=\"records\" returns the results as a list of dictionaries.

Verify installation
^^^^^^^^^^^^^^^^^^^

To ensure that werpy was installed successfully, you can check the installed version:

.. code-block:: console

   $ pip show werpy

This command should display information about the installed package, including the version number.


Usage
^^^^^

Now that werpy is installed, you can start using it in your Python projects. 
Refer to the documentation for examples and usage instructions.
