Normalization
===============

Normalizing reference and hypothesis texts is a crucial step in calculating the Word Error Rate (WER) in automatic speech recognition (ASR) or other natural language processing (NLP) tasks. 
Normalization involves standardizing the texts to ensure fair and accurate evaluation of the performance of the system.

werpy employs a set of normalization procedures, which encompass the following actions:

- The removal of all punctuation from the text.
- The elimination of any redundant or unnecessary whitespace, ensuring a streamlined and consistent presentation.
- The conversion of the entire text to lowercase, promoting uniformity and mitigating the impact of variations in letter case.

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


Advantages of Normalizing text 
""""""""""""""""""""""""""""""

There are are many benefits to normalizing reference and hypothesis texts when calculating the Word Error Rate:

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

- Handling Word Contractions and Expansions:
   Normalization addresses issues related to word contractions (e.g., "can't" vs. "cannot") and expansions (e.g., "it's" vs. "it is"). 
   Standardizing such variations ensures that the evaluation accurately reflects the similarity between the reference and hypothesis.

- Enhancing Metric Interpretability:
    Normalization enhances the interpretability of the evaluation metric. 
    It ensures that differences in the WER are primarily reflective of errors in content rather than variations in formatting or presentation.

Overall, normalizing reference and hypothesis texts is essential for ensuring a fair, consistent, and meaningful evaluation of ASR and NLP systems using metrics like Word Error Rate. 
It contributes to the reliability of the evaluation process and facilitates accurate comparisons between different systems.
