"""
Pytest configuration for web app tests.

Adds the web directory to sys.path so imports work correctly
whether running tests from project root or web directory.
"""

import sys
from pathlib import Path

# Add the web directory to sys.path for proper imports
web_dir = Path(__file__).parent.parent.parent
if str(web_dir) not in sys.path:
    sys.path.insert(0, str(web_dir))
