# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "werpy"
copyright = "Copyright 2023, Analytics in Motion"
author = "Ross Armstrong"
release = "2.1.1"

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
    "footer_links": ",".join(
        [
            "Documentation|https://werpy.readthedocs.io",
            "Package|https://pypi.org/project/werpy/",
            "Repository|https://github.com/analyticsinmotion/werpy",
            "Issues|https://github.com/analyticsinmotion/werpy/issues",
        ]
    ),
    # "show_powered_by": False,
}

html_static_path = ["_static"]
