Modules
=======

Modules available in werpy
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following table provides an overview of the modules that can be used in werpy:

.. list-table::
   :widths: 18 32
   :header-rows: 1

   * - Module
     - Description
   * - normalize(text)
     - Preprocess input text to remove punctuation, remove duplicated spaces, leading/trailing blanks and convert all words to lowercase.
   * - wer(reference, hypothesis) 
     - Calculate the overall Word Error Rate for the entire reference and hypothesis texts.
   * - wers(reference, hypothesis)
     - Calculates a list of the Word Error Rates for each of the reference and hypothesis texts.
   * - werp(reference, hypothesis)
     - Calculates a weighted Word Error Rate for the entire reference and hypothesis texts.
   * - werps(reference, hypothesis)
     - Calculates a list of weighted Word Error Rates for each of the reference and hypothesis texts.
   * - summary(reference, hypothesis)
     - Provides a comprehensive breakdown of the calculated results including the WER, Levenshtein Distance and all the insertion, deletion and substitution errors.
   * - summaryp(reference, hypothesis)
     - Delivers an in-depth breakdown of the results, covering metrics like WER, Levenshtein Distance, and a detailed account of insertion, deletion, and substitution errors, inclusive of the weighted WER.



Refer to the documentation for specific examples and usage instructions for these modules.
