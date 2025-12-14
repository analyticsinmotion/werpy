# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
The werpy package provides tools for calculating word error rates (WERs) and related metrics on text data.
"""

__version__ = "3.1.1"

from .errorhandler import error_handler
from .normalize import normalize
from .metrics import metrics
from .wer import wer
from .wers import wers
from .werp import werp
from .werps import werps
from .summary import summary
from .summaryp import summaryp

__all__ = [
    "error_handler",
    "normalize",
    "metrics",
    "wer",
    "wers",
    "werp",
    "werps",
    "summary",
    "summaryp",
]
