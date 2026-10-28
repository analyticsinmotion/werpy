# SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com>
# SPDX-License-Identifier: BSD-3-Clause

"""
The werpy package provides tools for calculating word error rates (WERs) and related metrics on text data.
"""

__version__ = "3.5.0"

from ._normalize import normalize
from ._wer import wer
from ._wers import wers
from ._werp import werp
from ._werps import werps
from ._summary import summary
from ._summaryp import summaryp

__all__ = [
    "normalize",
    "wer",
    "wers",
    "werp",
    "werps",
    "summary",
    "summaryp",
]
