Normalization
===============

Normalizing reference and hypothesis texts is a crucial step in calculating the Word Error Rate (WER) in automatic speech recognition (ASR) or other natural language processing (NLP) tasks.
Normalization involves standardizing the texts to ensure fair and accurate evaluation of the performance of the system.

werpy.normalize() applies the same rule to each text, in this order:

1. The text is put into Unicode Normalization Form C (NFC). A character that is written as one code point and the same character written as a letter followed by combining marks become the same text.
2. NUL, the ASCII punctuation characters of Python's string.punctuation and every character of the Unicode general category for punctuation are removed. The characters on each side of a removed character join, so well-known becomes wellknown.
3. The text is converted to lowercase. A removed punctuation character still ends a word for the Greek capital sigma, which becomes the final sigma before it.
4. The text is put into NFC again, because a removed character or the lowercase can let characters compose.
5. Every run of whitespace becomes one space, and the whitespace at the start and at the end is removed.

The following table shows what happens to each kind of character:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Character
     - Result
   * - Letters, and the other characters that have a lowercase form, such as the Roman numerals
     - Converted to lowercase.
   * - Letters with accents
     - Put into the composed form, so é as one character and e followed by a combining acute accent give the same result.
   * - The ASCII punctuation characters ``!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~``
     - Removed. The characters on each side join, so well-known becomes wellknown.
   * - Other punctuation, such as ’ “ ” — – … « » ¿ ¡ 、 。 ，
     - Removed in the same way.
   * - Symbols other than ASCII, such as € © ™ °, and emoji
     - Kept.
   * - Digits and numbers
     - Kept.
   * - Spaces, tabs, line breaks and the other characters for which str.isspace() is true
     - Each run becomes one space. The whitespace at the start and at the end is removed.
   * - Invisible format characters, such as the zero width space and the soft hyphen
     - Kept.
   * - NUL
     - Removed.
   * - Lone surrogates
     - Kept.
   * - Every other character, such as the other control characters and the private use characters
     - Kept.

normalize() is idempotent: a result that is normalized again does not change.

The rule uses the Unicode database of the Python that runs it. A text whose characters are all assigned in the Unicode version of the oldest Python that werpy supports gives the same result on every supported Python. A character that only a later version of Unicode assigns, such as a new punctuation character or a new combining mark, can give a different result on an older Python.

Normalize a list of text
------------------------

*Python Code*

.. code-block:: python

   import werpy

   input_data = ["It's very popular in Antarctica.", "The Sugar Bear character"]
   reference = werpy.normalize(input_data)
   print(reference)

*Results Output*

.. code-block:: python

   ['its very popular in antarctica', 'the sugar bear character']


Normalize text with Unicode punctuation and accents
---------------------------------------------------

The reference below is typed with a curly apostrophe, curly quotation marks and an em dash, and its é is the letter e followed by a combining acute accent. The hypothesis uses ASCII punctuation and é as one character. After normalize(), the two texts are the same, and their Word Error Rate is 0.0.

*Python Code*

.. code-block:: python

   import werpy

   reference = werpy.normalize("It’s “done” — cafe\N{COMBINING ACUTE ACCENT}")
   hypothesis = werpy.normalize("It's \"done\" - café")
   print(reference)
   print(werpy.wer(reference, hypothesis))

*Results Output*

.. code-block:: python

   its done café
   0.0


Advantages of Normalizing text
""""""""""""""""""""""""""""""

There are many benefits to normalizing reference and hypothesis texts when calculating the Word Error Rate:

- Consistency in Evaluation:
   Normalizing helps in maintaining consistency during the evaluation process.
   By applying standard rules to both the reference and hypothesis texts, you ensure that the evaluation is fair and unbiased.

- Robustness to Minor Variations:
   Normalization helps make the evaluation robust to minor variations in the transcriptions.
   Different transcriptions of the same content may have variations in punctuation, capitalization, or other non-semantic elements.
   Normalizing these aspects ensures that the evaluation focuses on the semantic content.

- Handling Case Sensitivity:
   Many evaluation metrics, including WER, are case-sensitive.
   Normalizing text to a consistent case (usually lowercase) helps in avoiding discrepancies in evaluation due to differences in capitalization.

- Punctuation Standardization:
   In spoken language, there can be variations in the use of punctuation.
   Normalizing punctuation ensures that the evaluation is not influenced by differences in how speakers use commas, periods, or other punctuation marks.

- Enhancing Metric Interpretability:
    Normalization enhances the interpretability of the evaluation metric.
    It ensures that differences in the WER are primarily reflective of errors in content rather than variations in formatting or presentation.

Overall, normalizing reference and hypothesis texts is essential for ensuring a fair, consistent, and meaningful evaluation of ASR and NLP systems using metrics like Word Error Rate.
It contributes to the reliability of the evaluation process and facilitates accurate comparisons between different systems.
