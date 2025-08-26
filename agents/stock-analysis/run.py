#!/usr/bin/env python3
"""
Stock Analysis Agent Runner Script
Configurable startup script for the Stock Analysis Agent with port management
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run the Stock Analysis Agent')
    parser.add_argument('--port', type=int, default=8513, 
                       help='Port to run the Streamlit app on (default: 8513)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host to run the Streamlit app on (default: localhost)')
    parser.add_argument('--dev', action='store_true',
                       help='Run in development mode with auto-reload')
    
    args = parser.parse_args()
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    app_path = script_dir / "app.py"
    
    if not app_path.exists():
        print(f"❌ Error: app.py not found at {app_path}")
        sys.exit(1)
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Prepare streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true"
    ]
    
    if args.dev:
        cmd.extend([
            "--server.runOnSave", "true",
            "--server.fileWatcherType", "auto"
        ])
    
    print(f"🚀 Starting Stock Analysis Agent on {args.host}:{args.port}")
    print(f"📂 Working directory: {script_dir}")
    print(f"🔗 Access URL: http://{args.host}:{args.port}")
    print(f"💡 Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Stock Analysis Agent stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Stock Analysis Agent: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()