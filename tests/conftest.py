"""Puts the POC root on sys.path, matching how every entry-point file
(app/streamlit_app.py, scripts/*.py) already does it — so `from retrieval...`/
`from citations...`/`from shared...` resolve under pytest regardless of the
invocation's working directory."""

import sys
from pathlib import Path

_POC_ROOT = Path(__file__).resolve().parent.parent
if str(_POC_ROOT) not in sys.path:
    sys.path.insert(0, str(_POC_ROOT))
