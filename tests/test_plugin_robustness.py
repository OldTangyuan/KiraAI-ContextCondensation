"""pytest entry point for the context_condensation plugin tests.

Run from the KiraAI repo root:

    python -m pytest data/plugins/context_condensation/tests/ -v

The canonical test functions live in ``self_test.py`` (which is also runnable
standalone without pytest). Re-exporting them here lets pytest discover them
under the conventional ``test_*.py`` name without duplicating the suite.
"""

import sys
from pathlib import Path

# Import the canonical suite whichever way pytest put this directory on the
# path (rootdir insertion makes it importable as a top-level module).
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

try:
    from .self_test import *  # package-style relative import (tests/ has __init__)
except ImportError:
    from self_test import *  # top-level import (rootdir insertion)

# Everything from self_test (test functions + _ALL_TESTS) is now in scope for
# pytest collection.
