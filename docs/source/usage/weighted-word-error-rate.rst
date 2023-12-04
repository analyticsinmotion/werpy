Weighted Word Error Rate
========================

werpy offers a feature for customizing the impact of different types of errors on the Word Error Rate (WER) calculation through the ability to add penalty weights. 
This is a useful capability as it recognizes that not all errors have the same significance or impact on the overall performance of the system. 
By allowing users to assign weights to different types of errors, werpy enables a more nuanced evaluation that aligns with the specific priorities and challenges of the given task or syntax being tested.


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


Summary
-------

Applying weights to Word Error Rates (WER) in automatic speech recognition and natural language processing evaluations confers several advantages and benefits to the assessment process. 

By assigning specific weights to different types of errors, such as insertions, deletions, and substitutions, the evaluation becomes more nuanced and context-aware. 
This customization allows practitioners to reflect the real-world impact of errors on the overall quality of system outputs.

Assigning varying weights acknowledges that not all errors are equal; some may be more disruptive or challenging to correct than others. 
This tailored approach enhances the fairness of evaluations, ensuring that the WER aligns more closely with the specific priorities and challenges of the linguistic or syntactic context being tested.

Overall, the application of weights to WER enhances the sensitivity of evaluations, contributing to a more accurate and insightful assessment of the performance of automatic speech recognition and natural language processing systems.
