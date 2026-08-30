"""Make ``src/`` importable for the test suite.

``pip install -e .`` normally handles this, but a repository should be testable
straight out of a clone -- and an editable install that has gone stale (a moved
checkout, a rebuilt virtualenv) otherwise fails at collection with an import
error that has nothing to do with the code.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
