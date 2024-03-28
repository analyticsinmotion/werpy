# CHANGELOG
This changelog file outlines a chronologically ordered list of the changes made on this project. 
It is organized by version and release date followed by a list of Enhancements, New Features, Bug Fixes, and/or Breaking Changes.
<br /><br />

## Version 2.1.2-dev (Dev)
**Released:** TBD<br />
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


<br /><br />

## Version 2.1.1 (Latest)
**Released:** November 27, 2023<br />
**Tag:** v2.1.1

### Enhancements

- Updated the meson.build file to align with the recommended approach for integrating Cython into the build process:
  - Added Cython to the list of languages utilized by the project.
  - Passed the Cython source code directly to the py.extension_module() definition for improved integration.
  - Specified the C standard configuration as C11, instructing Meson to use C11 as the designated C standard.


<br /><br />

## Version 2.1.0
**Released:** November 23, 2023<br />
**Tag:** v2.1.0

### New Feature

- Enhanced cross-platform support by integrating cibuildwheel, enabling compatibility with macOS and popular Linux distributions. With existing Windows compatibility, the package now spans all major configurations. Feel free to reach out if you have a specific OS configuration you'd like to discuss for potential inclusion.


<br /><br />

## Version 2.0.0 
**Released:** November 23, 2023<br />
**Tag:** v2.0.0

### New Feature

- Refactored the 'metrics' module to run on C instead of Python to enhance performance significantly. The pure Python code in metrics.py had (arguably) reached its optimization limit, with the dynamic programming module serving as the primary component for calculations. By migrating this critical module to C, the metrics module now operates 5x faster, resulting in a substantial improvement in application speed. The metrics calculations are now performed at the C level while retaining the original Python interface. All other modules will remain as python.

- During the transition to utilizing C optimizations, we opted to switch our Python package build system from Hatchling to Mesonpy. Mesonpy facilitates seamless compilation of C code as an integral part of the package build process. As a result of this transition, you can expect modifications to the pyproject.toml file and the introduction of a new meson.build file. This change under the hood enables us to integrate both Python and C code within the package natively for the performance optimizations.


### Breaking Changes

- In this significant application update, we are introducing phased support for different operating systems. Initially, this version will exclusively support Windows. However, swift additions for UNIX/Linux and macOS compatibility are already in the pipeline and will be incorporated promptly. This temporary change allows us to roll out the major version upgrade incrementally while ensuring reliability for our user base. 

- Certain web applications relying exclusively on pure Python environments might encounter challenges running this package successfully. If your applications are affected, please don't hesitate to get in touch to discuss potential compatibility issues.
 

<br /><br />

## Version 1.1.2
**Released:** November 17, 2023<br />
**Tag:** v1.1.2

### New Feature

- Implemented a new feature, summaryp, that furnishes a comprehensive analysis of outcomes. This includes detailed metrics such as Word Error Rate (WER), Levenshtein Distance, and a thorough report on insertion, deletion, and substitution errors. Most notably, the update includes the incorporation of weighted WER calculations for a more detailed assessment when adding penalties to the various error types.

- Added corresponding new tests for the 'summaryp' function within the 'werpy' package, enhancing test coverage and ensuring robust functionality. The additional tests provide comprehensive validation of the 'summaryp' function, contributing to improved reliability and accuracy in the package's performance.


### Bug Fixes

- Fixed an AttributeError in the 'summary.py' module that occurred when attempting to access the 'size' attribute of a 'float' object. This error happened when the module was provided a single reference and hypothesis string as input. The issue has been resolved.

- Fixed an issue with the attributes of DataFrame column name "ld". Resolved the discrepancy in the "dtype" attribute from int32 to int64.
 

<br /><br />

## Version 1.1.1 
**Released:** November 13, 2023<br />
**Tag:** v1.1.1

### Enhancements

- Integrated CircleCI with GitHub repository for automated unit testing. Configured CircleCI workflow to run tests on each commit and pull request. This allows for continuous inspection of code quality and testing.

- Added the CircleCI badge to the repository Readme.md file


### Bug Fixes

- Fixed an AttributeError in the wers.py module caused by an non-standard operation on a 'float' object. This only occurred when a single reference and hypothesis input string was entered and has now been rectified.

- Resolved an AttributeError in the 'werps.py' module, which was triggered by a 'float' object having a size attribute. This issue specifically arose when a single reference and hypothesis input string was provided, and it has been successfully addressed.
 

<br /><br />

## Version 1.1.0
**Released:** November 8, 2023<br />
**Tag:** v1.1.0

### Enhancements

- In this incremental version update, we've optimized the distance algorithm calculations to improve their processing speed. Users should experience improved performance without encountering any disruptive alterations.

- Added the following unit tests to improve code coverage and validation for the functions in the werpy module. The new tests cover additional use cases with longer input sequences and help ensure the wer calculation works properly in different scenarios.
  - Added new unit tests for the wer module.
  - Added new unit tests for the wers module. 
  - Added new unit tests for the werp module. 
  - Added new unit tests for the werps module. 
  - Added new unit tests for the summary module.  

<br /><br />

## Version 1.0.0
**Released:** November 2, 2023<br />
**Tag:** v1.0.0

### Enhancements

- In this major version update, the normalize.py module has undergone a comprehensive refactor to enhance its performance and streamline its codebase. These changes represent a significant leap in the efficiency and readability of the code. It has been optimized to process data at a significantly faster pace, reducing processing time for various tasks, and with cleaner, more maintainable code. These enhancements mark a significant milestone in our development, paving the way for better performance, increased efficiency, and improved code quality.

- Added new unit tests for the normalize module. These tests focus on improving test coverage, enhancing the reliability of the module, and ensuring the accuracy of the normalization process. By incorporating these tests, we aim to identify and address issues early in the development cycle, making the upcoming release more stable and reliable.

<br /><br />

## Version 0.0.5
**Released:** October 26, 2023<br />
**Tag:** v0.0.5

### Enhancements

- Removed two methods from the "normalization" module:
  - `remove_leading_trailing_spaces`: This method was used to remove leading and trailing spaces in the input text. (Method removed in this release)
  - `replace_multiple_spaces`: This method was used to convert consecutive double spaces into single spaces in the input text. (Method removed in this release)

- Added a new method to the "normalization" module:
  - `remove_whitespace(text)`: This new method efficiently removes all excess spaces in the input text. It replaces consecutive sequences of spaces with a single space and removes any leading or trailing spaces, ensuring a cleaner and more consistent text output.

<br /><br />

## Version 0.0.4 
**Released:** May 4, 2023<br />
**Tag:** v0.0.4

### Enhancements

- The code to handle exceptions and errors has been refactored to reduce code duplication across modules. In addition, the changes will make adding and testing errors or exceptions easier to maintain in the future.


### Bug Fix

- Fixed a number of inconsistent return statements (R1710) within the package modules. This ensures that all functions will return a consistent expression when called.

<br /><br />

## Version 0.0.3
**Released:** May 2, 2023<br />
**Tag:** v0.0.3

### Bug Fix

- Fixed a bug contained within the modules that was causing a Cyclic Import issue (R0401). One of the import statements was missing a period at the start of the module name. The fix has been tested and deployed successfully.

<br /><br />

## Version 0.0.2 
**Released:** May 1, 2023<br />
**Tag:** v0.0.2

### General Changes

- Added Module Docstrings


### Bug Fix

- Fixed an unidiomatic-typecheck (C0123) from type() to isinstance(). The idiomatic way to perform an explicit typecheck in Python is to use isinstance(x, y) rather than type(x) == Y.

<br /><br />

## Version 0.0.1 (Initial Release)
**Released:** April 28, 2023<br />
**Tag:** v0.0.1

This is the initial release
