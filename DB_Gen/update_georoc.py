#!/usr/bin/env python3
"""
Backward-compatible entry point for update_georoc.py.
Installs georocdata package if needed, then delegates to it.
"""

try:
    from georocdata.cli import main
    main()
except ImportError:
    import subprocess
    import sys
    print("Installing georocdata package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    from georocdata.cli import main
    main()