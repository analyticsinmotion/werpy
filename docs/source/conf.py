# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
This is the configuration file for the Sphinx documentation builder.

It contains project information and various settings for building the docs.
"""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

PROJECT = "werpy"
PROJECT_COPYRIGHT = "2023 Analytics in Motion"
AUTHOR = "Ross Armstrong"
RELEASE = "3.0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
HTML_THEME = "sphinx_nefertiti"
html_theme_options = {
    # ... other options ...
    "repository_name": "analyticsinmotion/werpy",
    "repository_url": "https://github.com/analyticsinmotion/werpy",
    "footer_links": [
        {
            "text": "Documentation", 
            "link": "https://werpy.readthedocs.io"
        },
        {
            "text": "Package", 
            "link": "https://pypi.org/project/werpy/"
        },
        {
            "text": "Repository", 
            "link": "https://github.com/analyticsinmotion/werpy"
        },
        {
            "text": "Issues", 
            "link": "https://github.com/analyticsinmotion/werpy/issues"
        },
    ],
    # "show_powered_by": False,
}

html_static_path = ["_static"]
