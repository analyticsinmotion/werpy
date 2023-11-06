# CHANGELOG
This changelog file outlines a chronologically ordered list of the changes made on this project. 
It is organized by version and release date followed by a list of Enhancements, New Features, Bug Fixes, and/or Breaking Changes.
<br /><br />


## Version 1.0.1-dev1 (Under Development)
**Released:** TBD<br />
**Tag:** v1.0.1

### Enhancements

- Added the following unit tests to improve code coverage and validation for the functions in the werpy module. The new tests cover additional use cases with longer input sequences and help ensure the wer calculation works properly in different scenarios.
  - Added new unit tests for the wer module.
  - Added new unit tests for the wers module. 
  - Added new unit tests for the werp module. 
  - Added new unit tests for the werps module. 
  - Added new unit tests for the summary module.  

<br /><br />

## Version 1.0.0 (Latest)
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
