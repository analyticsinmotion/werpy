# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "werpy"
copyright = "Copyright 2023-2025, Analytics in Motion"
author = "Ross Armstrong"
release = "3.0.0"

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
