#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    port = int(os.environ.get("AGENT_PORT", 8512))
    streamlit_app = Path(__file__).parent / "ui" / "streamlit_app.py"
    os.system(f"streamlit run {streamlit_app} --server.port {port} --server.address 0.0.0.0")