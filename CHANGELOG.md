<!-- markdownlint-disable MD024 -->
# CHANGELOG

This changelog file outlines a chronologically ordered list of the changes made on this project.
It is organized by version and release date followed by a list of Enhancements, New Features, Bug Fixes, and/or Breaking Changes.

## Version 3.1.1

**Released:** December 14, 2025  
**Tag:** v3.1.1

### Bug Fixes

- Fixed IndexError in wer, wers, werp, werps, summary, and summaryp functions when processing single string inputs. The issue occurred due to incorrect handling of np.vectorize return values, which produce different array structures for single versus batch inputs (0-dimensional arrays for single strings, 1-dimensional object arrays for lists). The fix unwraps 0-dimensional arrays and uses element-type checking to distinguish between single example vectors and batch processing scenarios.

- Improved handling of ragged data structures in batch processing by implementing direct field summation instead of transpose operations. This prevents errors when processing mixed scalar and list data in the word error rate breakdown results.

### Enhancements

- Added comprehensive benchmarking infrastructure to compare performance against werx and jiwer packages using the LibriSpeech evaluation dataset.

- Added optional dependencies groups in pyproject.toml for testing and benchmarking workflows, enabling easier development environment setup.

## Version 3.1.0

**Released:** April 22, 2025  
**Tag:** v3.1.0

### Enhancements

- Updated the 'summary' function in 'summary.py' to include a precise return type hint: 'pd.DataFrame | None'.
  - Improved readability and type safety for the function.

- Updated the return type hint for the 'wer' function in 'wer.py' from 'float' to 'float | np.float64 | None'.
  - Enhanced type accuracy and alignment with the function's behavior.

- Continued work on type hinting improvements and resolving 'mypy' errors for better code quality and maintainability.

- Optimized Levenshtein distance matrix initialization in calculations() by replacing a Python list-of-lists with a Cython-typed NumPy array (cdef int[:, :]). This reduces memory overhead and significantly speeds up execution on typical workloads, especially for large datasets or repeated function calls. It improves scalability, responsiveness, and memory efficiency.

- Refactored internal variable typing in calculations() for clarity and consistency:
  - Loop indices and size variables now use Py_ssize_t, matching Python's internal conventions.
  - Grouped and explicitly typed intermediate variables like inserted_words, deleted_words, and substituted_words for improved readability and static checks. This enhances code quality, reduces reliance on dynamic typing in performance-critical paths, and prepares the function for future optimizations.

## Version 3.0.2

**Released:** April 3, 2025  
**Tag:** v3.0.2

### Enhancements

- Updated tests to better handle and validate invalid input types.

- Improved code quality by addressing some minor CodeFactor alerts:
  - Fixed naming issues and built-in name overrides.
  - Updated constants to use proper UPPER_CASE naming style.

- Included 'ZeroDivisionError' in exception handling across source files for better error coverage.

- Adopted REUSE and SDCX specifications for license compliance:
  - Removed unnecessary license files for third-party dependencies ('numpy', 'pandas', 'cython') from the 'LICENSES' directory.
  - Added 'BSD-3-Clause.txt' license file to the 'LICENSES' directory.
  - Added 'SPDX-FileCopyrightText' and 'SPDX-License-Identifier' headers directly in all '.py' and '.pyx' source files.
  - Created a 'REUSE.toml' file to centralize license declarations. The benefits of doing this for other files include:
    - Centralized license declarations — avoids repetitive SPDX headers in every file (there are lots of other files).
    - Cleaner source files — especially for binary or non-comment-friendly formats (e.g., '.png', '.yml', '.toml').
    - Scalable and maintainable — simplifies licensing for new files and directories.
    - Improved automation — supports tools like 'reuse lint' and CI-based license checks.
    - Clarity and compliance — enhances transparency and ensures open source compliance through machine-readable metadata.

- Initial support for static type checking: included py.typed marker file to enable type checkers to recognize the package as typed. Note: full type coverage is not yet guaranteed and will be improved incrementally.

## Version 3.0.1

**Released:** March 26, 2025  
**Tag:** v3.0.1

### Enhancements

- Custom exception handling for ZeroDivisionError to catch cases where the reference is blank or both reference and hypothesis are empty. When either of these events occur a clear, descriptive error message is Raised:
"Invalid input: reference must not be blank, and reference and hypothesis cannot both be empty."

- Unit test added to verify that blank reference and hypothesis inputs raise a ZeroDivisionError.

- Publishing process is now automated using GitHub Actions and PyPI Trusted Publishing

## Version 3.0.0

**Released:** March 20, 2025  
**Tag:** v3.0.0

### Breaking Changes

- Dropped support for Python 3.8.x and 3.9.x
  - These versions have reached or are nearing their end of life
  - Users are strongly encouraged to upgrade to Python 3.10 or later
  - This change allows us to utilize newer Python features, improve overall package maintenance and security
  - Minimum supported Python version is now 3.10

- Dropped support for Pandas 1.x (pandas>=2.0.0 is now required)

- NumPy versions below 1.26.0 are no longer supported for Python 3.10+

- NumPy 2.x is required for Python 3.12+, which may introduce API changes

### Enhancements

- Official support for Python 3.13:
  - This release ensures full compatibility with Python 3.13, with thorough testing to verify seamless functionality.
  - All dependencies have been reviewed and confirmed to be compatible with Python 3.13.

- Updated minimum required versions for NumPy dependencies:
  - For Python < 3.12: Increased from 1.21.6 to 1.26.0
  - For Python >= 3.12: Increased to 2.2.0 (previously 1.23.2 for Python >= 3.11)

- Updated minimum required versions for Pandas dependencies:
  - Increased from 1.3.0 to 2.0.0, ensuring compatibility with the latest NumPy versions.

- Bump sphinx from 7.3.7 to 8.1.3

- Bump sphinx-nefertiti from 0.3.4 to 0.7.4
  - The latest version ('0.7.4') is now fully compatible with Sphinx 8.1.

- Updated C standard from C11 ('c_std=c11') to C17 ('c_std=c17') in the Meson build system.
  - Improves compatibility with modern compilers.
  - Aligns with the latest stable C standard while maintaining backward compatibility.
  - No breaking changes expected, as C17 is a bug-fix refinement of C11.

- Updated Cython NumPy Import Convention
  - Changed cimport numpy as np to cimport numpy as cnp to align with Cython best practices, improving clarity between Python and Cython NumPy APIs.

- Explicit NumPy C API Initialization
  - Added cnp.import_array() to ensure proper initialization of NumPy’s C API, resolving the numpy.core.multiarray failed to import error in NumPy 2.x+.

- Updated Cython Function Type Annotation
  - Changed cpdef np.ndarray calculations(...) to cpdef cnp.ndarray calculations(...) to properly reference the Cython-level NumPy API, ensuring type safety and compatibility with compiled C extensions.

## Version 2.1.3-beta

**Released:** Not released  
**Tag:** v2.1.3-beta

### Enhancements

All the following changes were incorporated into the major v3.0.0 rollout

- Bump idna from 3.6 to 3.7 in /docs ([#5](https://github.com/analyticsinmotion/werpy/pull/5))
- Bump sphinx from 7.2.6 to 7.3.7 ([#6](https://github.com/analyticsinmotion/werpy/pull/6))
- Bump jinja2 from 3.1.3 to 3.1.4 in /docs ([#7](https://github.com/analyticsinmotion/werpy/pull/7))
- Bump requests from 2.31.0 to 2.32.2 in /docs ([#10](https://github.com/analyticsinmotion/werpy/pull/10))
- Bump urllib3 from 2.1.0 to 2.2.2 in /docs ([#11](https://github.com/analyticsinmotion/werpy/pull/11))
- Bump sphinx-nefertiti from 0.3.2 to 0.3.4 ([#12](https://github.com/analyticsinmotion/werpy/pull/12))
- Bump certifi from 2023.11.17 to 2024.7.4 in /docs ([#13](https://github.com/analyticsinmotion/werpy/pull/13))

## Version 2.1.2

**Released:** April 5, 2024  
**Tag:** v2.1.2

### Enhancements

- Full compatibility with Python v3.12:
  - The package has been thoroughly tested and verified to work seamlessly with Python v3.12.
  - All dependencies have been updated and are also compliant with Python v3.12.

- Added comprehensive package documentation using Sphinx and hosted on Read the Docs. The documentation covers installation instructions, usage guides, and worked examples of all feature functionality. Access the documentation at [https://werpy.readthedocs.io/](https://werpy.readthedocs.io/).

- Updated the circleci config.yml file to support the compilation of Cython .pyx files.

- Integrated codecov into the circleci config.yml file to generate comprehensive coverage reports.

- Enhanced testing procedures to increase code coverage percentage.

- Ensured compliance with the Black code formatting by modifying relevant files.

### Changed

- Bump jinja2 from 3.1.2 to 3.1.3 in /docs. ([#1](https://github.com/analyticsinmotion/werpy/pull/1))
  - For more information please see the Jinja release documentation [https://github.com/pallets/jinja/releases](https://github.com/pallets/jinja/releases).

- Bump sphinx-nefertiti from 0.2.1 to 0.2.3 ([#2](https://github.com/analyticsinmotion/werpy/pull/2))

- Bump sphinx-nefertiti from 0.2.3 to 0.3.1 ([#3](https://github.com/analyticsinmotion/werpy/pull/3))

## Version 2.1.1

**Released:** November 27, 2023  
**Tag:** v2.1.1

### Enhancements

- Updated the meson.build file to align with the recommended approach for integrating Cython into the build process:
  - Added Cython to the list of languages utilized by the project.
  - Passed the Cython source code directly to the py.extension_module() definition for improved integration.
  - Specified the C standard configuration as C11, instructing Meson to use C11 as the designated C standard.

## Version 2.1.0

**Released:** November 23, 2023  
**Tag:** v2.1.0

### New Feature

- Enhanced cross-platform support by integrating cibuildwheel, enabling compatibility with macOS and popular Linux distributions. With existing Windows compatibility, the package now spans all major configurations. Feel free to reach out if you have a specific OS configuration you'd like to discuss for potential inclusion.

## Version 2.0.0

**Released:** November 23, 2023  
**Tag:** v2.0.0

### New Feature

- Refactored the 'metrics' module to run on C instead of Python to enhance performance significantly. The pure Python code in metrics.py had (arguably) reached its optimization limit, with the dynamic programming module serving as the primary component for calculations. By migrating this critical module to C, the metrics module now operates 5x faster, resulting in a substantial improvement in application speed. The metrics calculations are now performed at the C level while retaining the original Python interface. All other modules will remain as python.

- During the transition to utilizing C optimizations, we opted to switch our Python package build system from Hatchling to Mesonpy. Mesonpy facilitates seamless compilation of C code as an integral part of the package build process. As a result of this transition, you can expect modifications to the pyproject.toml file and the introduction of a new meson.build file. This change under the hood enables us to integrate both Python and C code within the package natively for the performance optimizations.

### Breaking Changes

- In this significant application update, we are introducing phased support for different operating systems. Initially, this version will exclusively support Windows. However, swift additions for UNIX/Linux and macOS compatibility are already in the pipeline and will be incorporated promptly. This temporary change allows us to roll out the major version upgrade incrementally while ensuring reliability for our user base.

- Certain web applications relying exclusively on pure Python environments might encounter challenges running this package successfully. If your applications are affected, please don't hesitate to get in touch to discuss potential compatibility issues.

## Version 1.1.2

**Released:** November 17, 2023  
**Tag:** v1.1.2

### New Feature

- Implemented a new feature, summaryp, that furnishes a comprehensive analysis of outcomes. This includes detailed metrics such as Word Error Rate (WER), Levenshtein Distance, and a thorough report on insertion, deletion, and substitution errors. Most notably, the update includes the incorporation of weighted WER calculations for a more detailed assessment when adding penalties to the various error types.

- Added corresponding new tests for the 'summaryp' function within the 'werpy' package, enhancing test coverage and ensuring robust functionality. The additional tests provide comprehensive validation of the 'summaryp' function, contributing to improved reliability and accuracy in the package's performance.

### Bug Fixes

- Fixed an AttributeError in the 'summary.py' module that occurred when attempting to access the 'size' attribute of a 'float' object. This error happened when the module was provided a single reference and hypothesis string as input. The issue has been resolved.

- Fixed an issue with the attributes of DataFrame column name "ld". Resolved the discrepancy in the "dtype" attribute from int32 to int64.

## Version 1.1.1

**Released:** November 13, 2023  
**Tag:** v1.1.1

### Enhancements

- Integrated CircleCI with GitHub repository for automated unit testing. Configured CircleCI workflow to run tests on each commit and pull request. This allows for continuous inspection of code quality and testing.

- Added the CircleCI badge to the repository Readme.md file

### Bug Fixes

- Fixed an AttributeError in the wers.py module caused by an non-standard operation on a 'float' object. This only occurred when a single reference and hypothesis input string was entered and has now been rectified.

- Resolved an AttributeError in the 'werps.py' module, which was triggered by a 'float' object having a size attribute. This issue specifically arose when a single reference and hypothesis input string was provided, and it has been successfully addressed.

## Version 1.1.0

**Released:** November 8, 2023  
**Tag:** v1.1.0

### Enhancements

- In this incremental version update, we've optimized the distance algorithm calculations to improve their processing speed. Users should experience improved performance without encountering any disruptive alterations.

- Added the following unit tests to improve code coverage and validation for the functions in the werpy module. The new tests cover additional use cases with longer input sequences and help ensure the wer calculation works properly in different scenarios.
  - Added new unit tests for the wer module.
  - Added new unit tests for the wers module.
  - Added new unit tests for the werp module.
  - Added new unit tests for the werps module.
  - Added new unit tests for the summary module.  

## Version 1.0.0

**Released:** November 2, 2023  
**Tag:** v1.0.0

### Enhancements

- In this major version update, the normalize.py module has undergone a comprehensive refactor to enhance its performance and streamline its codebase. These changes represent a significant leap in the efficiency and readability of the code. It has been optimized to process data at a significantly faster pace, reducing processing time for various tasks, and with cleaner, more maintainable code. These enhancements mark a significant milestone in our development, paving the way for better performance, increased efficiency, and improved code quality.

- Added new unit tests for the normalize module. These tests focus on improving test coverage, enhancing the reliability of the module, and ensuring the accuracy of the normalization process. By incorporating these tests, we aim to identify and address issues early in the development cycle, making the upcoming release more stable and reliable.

## Version 0.0.5

**Released:** October 26, 2023  
**Tag:** v0.0.5

### Enhancements

- Removed two methods from the "normalization" module:
  - `remove_leading_trailing_spaces`: This method was used to remove leading and trailing spaces in the input text. (Method removed in this release)
  - `replace_multiple_spaces`: This method was used to convert consecutive double spaces into single spaces in the input text. (Method removed in this release)

- Added a new method to the "normalization" module:
  - `remove_whitespace(text)`: This new method efficiently removes all excess spaces in the input text. It replaces consecutive sequences of spaces with a single space and removes any leading or trailing spaces, ensuring a cleaner and more consistent text output.

## Version 0.0.4

**Released:** May 4, 2023  
**Tag:** v0.0.4

### Enhancements

- The code to handle exceptions and errors has been refactored to reduce code duplication across modules. In addition, the changes will make adding and testing errors or exceptions easier to maintain in the future.

### Bug Fix

- Fixed a number of inconsistent return statements (R1710) within the package modules. This ensures that all functions will return a consistent expression when called.

## Version 0.0.3

**Released:** May 2, 2023  
**Tag:** v0.0.3

### Bug Fix

- Fixed a bug contained within the modules that was causing a Cyclic Import issue (R0401). One of the import statements was missing a period at the start of the module name. The fix has been tested and deployed successfully.

## Version 0.0.2

**Released:** May 1, 2023  
**Tag:** v0.0.2

### General Changes

- Added Module Docstrings

### Bug Fix

- Fixed an unidiomatic-typecheck (C0123) from type() to isinstance(). The idiomatic way to perform an explicit typecheck in Python is to use isinstance(x, y) rather than type(x) == Y.

## Version 0.0.1 (Initial Release)

**Released:** April 28, 2023  
**Tag:** v0.0.1

This is the initial release
