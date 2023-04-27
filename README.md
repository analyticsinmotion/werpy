# werpy - Word Error Rate for Python

<!-- badges: start -->
[![Python Version](https://img.shields.io/badge/python-3.8%7C3.9%7C3.10%7C3.11-blue?logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)&nbsp;&nbsp;
[![werpy License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/analyticsinmotion/werpy/blob/main/LICENSE)&nbsp;&nbsp;
![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)&nbsp;&nbsp;
<!-- badges: end -->

## What is it?
**werpy** is a powerful yet lightweight Python package that rapidly calculates and analyzes the Word Error Rate (WER) between two sets of text. 
It has been designed with the flexibility to handle multiple input data types such as strings, lists and numpy arrays.<br />

The package also includes a full set of features such as normalizing the input text to account for data collection variability and the capability to easily assign different weights/penalties to specific error classifications (insertions, deletions, and substitutions).
Additionally, the summary function provides a comprehensive breakdown of the calculated results to assist in analyzing the specific errors quickly and in more detail.
<br />

## Installation
You can install the latest **werpy** release with Python's pip package manager:

```python
# install werpy via PyPi
pip install werpy
```

<br /><br />
## Licensing

``werpy`` is released under the terms of the BSD 3-Clause License. Please refer to the LICENSE file for full details.

This project also includes third-party packages distributed under the BSD-3-Clause license, including NumPy and Pandas.

The full NumPy and Pandas licenses can be found in the <a href="https://github.com/analyticsinmotion/werpy/tree/main/LICENSES">LICENSES</a> directory in this repository. 

They can also be found directly in the following source codes:

 - Numpy - <a href="https://github.com/numpy/numpy/blob/main/LICENSE.txt">https://github.com/numpy/numpy/blob/main/LICENSE.txt</a>
 - Pandas - <a href="https://github.com/pandas-dev/pandas/blob/main/LICENSE">https://github.com/pandas-dev/pandas/blob/main/LICENSE</a>

