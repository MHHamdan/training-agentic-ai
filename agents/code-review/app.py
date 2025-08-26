"""
Main Application Entry Point for Code Review Agent
Enterprise-grade AI code review with multi-provider support
Author: Mohammed Hamdan
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.streamlit_app import main

if __name__ == "__main__":
    main()