# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Whether a model can create a working application from a written specification.

One large question, asked of as many models as you point it at: given a spec for
an installable package with a JSON API, a storage layer, a web page and its own
tests, does what the model writes run, and how much of the contract does it meet.

The answer is close to a single bit -- it runs or it does not -- which is too
coarse to grade a model with, and precisely why this is its own tool. What it
leaves behind is an application you can open and run, which is the point as much
as the score is.
"""
__all__ = ["SPEC", "TASK"]
__version__ = "1.0.0"

from .task import SPEC, TASK
