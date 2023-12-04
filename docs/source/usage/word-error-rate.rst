Word Error Rate
===============

The Word Error Rate (WER) is an evaluation metric commonly used in automatic speech recognition (ASR) and natural language processing (NLP) tasks. 
It quantifies the dissimilarity between a system-generated transcript (hypothesis) and a reference transcript. 

WER is calculated by measuring the minimum number of insertions, deletions, and substitutions required to transform the hypothesis into the reference, normalized by the total number of words in the reference. 
A lower WER indicates better accuracy, reflecting the system's ability to transcribe spoken or written language more closely to the provided reference. 



Simple Word Error Rate Calculation
----------------------------------

*Python Code*

.. code-block:: python

   import werpy
   
   wer = werpy.wer('i love cold pizza', 'i love pizza')
   print(wer)

*Results Output*

.. code-block:: python

   0.25

|

There are two main types of Word Error Rates (WER) commonly used in evaluation - **Cumulative** and **Sequence-level** Word Error Rates. 
The Simple Word Error Rate Calculation above is a basic example of a Cumulative Word Error Rate.

Cumulative Word Error Rate
--------------------------

Cumulative WER calculates the total WER across all sequences collectively. 
It considers the cumulative number of insertions, deletions, and substitutions over the entire dataset, providing a comprehensive measure of the system's overall performance. 
Cumulative WER is beneficial when the goal is to assess the system's global accuracy across a diverse set of inputs, emphasizing the aggregate error rate rather than individual sequence variations.


*Python Code*

.. code-block:: python

   import werpy
   
   ref = ['i love cold pizza','the sugar bear character was popular']
   hyp = ['i love pizza','the sugar bare character was popular']
   wer = werpy.wer(ref, hyp)
   print(wer)

*Results Output*

.. code-block:: python

   0.2

*Result Interpretation*

A cumulative Word Error Rate of 0.2 across two sequences of references and hypotheses indicates an overall error rate of 20% when considering both sequences together. 
This means that, on average, 20% of the words in the combined hypothesis transcripts differ from the reference transcripts across the two sequences.


Sequence-level Word Error Rates
-------------------------------

Sequence-level WER calculates and returns an individual WER for each pair of reference and hypothesis sequences. 
Each pair is evaluated independently, and the resulting WER reflects the error rate specific to that particular sequence. 
This approach is useful when there is a need to analyze the performance on a per-sequence basis, providing insights into how well the system performs across different inputs.

*Python Code*

.. code-block:: python

   import werpy
   
   ref = ['no one else could claim that','she cited multiple reasons why']
   hyp = ['no one else could claim that','she sighted multiple reasons why']
   wers = werpy.wers(ref, hyp)
   print(wers)

*Results Output*

.. code-block:: python

   [0.0, 0.2]

*Result Interpretation*

- The WER of 0.0 for the first sequence indicates that there are no errors between the reference and hypothesis transcripts for this specific sequence. The system's transcription perfectly matches the reference in this case.

- The WER of 0.2 for the second sequence indicates that there is a 20% error rate between the reference and hypothesis transcripts for this specific sequence. This means that, on average, 20% of the words in the hypothesis transcript differ from the reference transcript.


Summary
-------

While Sequence-level WER allows for a more granular analysis of performance on a per-sequence basis, Cumulative WER provides a holistic measure that considers the overall accuracy across all sequences in the evaluation dataset. 
The choice between these two types depends on the specific goals and requirements of the evaluation task.
