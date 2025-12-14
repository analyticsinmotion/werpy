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

from datetime import datetime

project = "werpy"
copyright = f'{datetime.now().year} <a href="https://www.analyticsinmotion.com">Analytics in Motion</a>'
author = "Ross Armstrong"
release = "3.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_nefertiti"
html_theme_options = {
    "header_links": [
        {
            "text": "Home",
            "link": "index",
        },
        {
            "text": "Installation",
            "link": "installation",
        },
        {
            "text": "Modules",
            "link": "modules",
        },
        {
            "text": "Usage",
            "dropdown": [
                {
                    "text": "Normalization",
                    "link": "usage/normalization",
                },
                {
                    "text": "Word Error Rate",
                    "link": "usage/word-error-rate",
                },
                {
                    "text": "Weighted Word Error Rate",
                    "link": "usage/weighted-word-error-rate",
                },
                {
                    "text": "Summarization",
                    "link": "usage/summarization",
                },
            ],
        },
    ],
    "footer_links": [
        {
            "text": "Analytics in Motion",
            "link": "https://www.analyticsinmotion.com"
        },
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
}

html_static_path = ["_static"]
