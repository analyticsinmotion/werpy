Weighted Word Error Rate
========================

werpy offers a feature for customizing the impact of different types of errors on the Word Error Rate (WER) calculation through the ability to add penalty weights. 
This is a useful capability as it recognizes that not all errors have the same significance or impact on the overall performance of the system. 
By allowing users to assign weights to different types of errors, werpy enables a more nuanced evaluation that aligns with the specific priorities and challenges of the given task or syntax being tested.

.. note::

   Each weight is a real number, such as an int or a float, that is finite and at least 0. All three weights may be 0, and the weighted Word Error Rate is then 0.0.


Cumulative Weighted Word Error Rate
-----------------------------------

Apply the following weights/penalties to the errors:

- Insertions: 0.5
- Deletions: 0.5
- Substitutions: 1

*Python Code*

.. code-block:: python

   import werpy
   
   ref = ['it was beautiful and sunny today','tomorrow may not be as nice']
   hyp = ['it was a beautiful and sunny day','tomorrow may not be as nice']
   werp = werpy.werp(ref, hyp, insertions_weight=0.5, deletions_weight=0.5, substitutions_weight=1)
   print(werp)

*Results Output*

.. code-block:: python

   0.125

*Result Interpretation*

- The weighted WER of 0.125 represents the error rate across two sequences of text after applying the specified weights. This is a measure of the system's performance that accounts for the varying impact of insertions, deletions, and substitutions based on the assigned weights.

- Insertions and Deletions are considered to have half the penalty of substitutions, which have a weight of 1. This reflects a specific prioritization where substitutions are considered to be more impactful or significant errors.



Sequence-level Weighted Word Error Rate
---------------------------------------

Apply the following weights/penalties to the errors:

- Insertions: 0.5
- Deletions: 0.5
- Substitutions: 1

*Python Code*

.. code-block:: python

   import werpy
   
   ref = ['it blocked sight lines of central park', 'her father was an alderman in the city government']
   hyp = ['it blocked sightlines of central park', 'our father was an elder man in the city government']
   werps = werpy.werps(ref, hyp, insertions_weight = 0.5, deletions_weight = 0.5, substitutions_weight = 1)
   print(werps)

*Results Output*

.. code-block:: python

   [0.21428571428571427, 0.2777777777777778]

*Result Interpretation*

- For the first sequence, the Weighted Word Error Rate is approximately 21.43%. This means that, after considering the assigned weights for insertions, deletions, and substitutions, about 21.43% of the words in the hypothesis transcript differ from the reference transcript in this sequence.

- For the second sequence, the Weighted Word Error Rate is approximately 27.78%. This indicates that, in the second sequence, about 27.78% of the words in the hypothesis transcript differ from the reference transcript, after applying the specified weights to different error types.

- Insertions and Deletions are considered to have half the penalty of substitutions. Substitutions are given a weight of 1, suggesting they are weighted at their full value.


How the errors are counted
--------------------------

werpy counts the errors on an alignment of the reference and hypothesis words with the fewest edits, as for the Word Error Rate.
Some pairs of texts have more than one such alignment, for example when two words are swapped.
werp(), werps() and summaryp() then use the alignment with the lowest weighted cost: the one with the fewest substitutions when a substitution weighs more than the mean of an insertion and a deletion, and the one with the most substitutions when it weighs less.

*Python Code*

.. code-block:: python

   import werpy

   ref = 'please call me back tomorrow'
   hyp = 'please call back me tomorrow'
   print(werpy.werp(ref, hyp, insertions_weight=0.5, deletions_weight=0.5, substitutions_weight=1))
   print(werpy.werp(ref, hyp, insertions_weight=2.5, deletions_weight=0.75, substitutions_weight=1.3))

*Results Output*

.. code-block:: python

   0.2
   0.52

*Result Interpretation*

- Two alignments have two edits each. One substitutes 'back' for 'me' and 'me' for 'back'. The other deletes 'me' and inserts 'me' after 'back'.

- With the weights 0.5, 0.5 and 1, the two substitutions cost 2 and the deletion and the insertion cost 1. werp() uses the deletion and the insertion, and the weighted WER is 1 / 5 = 0.2.

- With the weights 2.5, 0.75 and 1.3, the two substitutions cost 2.6 and the deletion and the insertion cost 3.25. werp() uses the two substitutions, and the weighted WER is 2.6 / 5 = 0.52.

- When a substitution weighs exactly the mean of an insertion and a deletion, for example when the three weights are equal, every alignment with the fewest edits has the same weighted cost. werp(), werps() and summaryp() then use the alignment of summary(), and summaryp() gives the same counts and word lists as summary().


Summary
-------

Applying weights to Word Error Rates (WER) in automatic speech recognition and natural language processing evaluations confers several advantages and benefits to the assessment process. 

By assigning specific weights to different types of errors, such as insertions, deletions, and substitutions, the evaluation becomes more nuanced and context-aware. 
This customization allows practitioners to reflect the real-world impact of errors on the overall quality of system outputs.

Assigning varying weights acknowledges that not all errors are equal; some may be more disruptive or challenging to correct than others. 
This tailored approach enhances the fairness of evaluations, ensuring that the WER aligns more closely with the specific priorities and challenges of the linguistic or syntactic context being tested.

Overall, the application of weights to WER enhances the sensitivity of evaluations, contributing to a more accurate and insightful assessment of the performance of automatic speech recognition and natural language processing systems.
