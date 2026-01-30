#!/usr/bin/env python3
import sys
import asyncio

try:
    from gui import InterviewGUI, run_cli, HAS_GUI
except ImportError:
    HAS_GUI = False
    from gui import run_cli

def main():
    cli_mode = "--cli" in sys.argv or "-c" in sys.argv
    if cli_mode or not HAS_GUI:
        asyncio.run(run_cli())
    else:
        InterviewGUI().run()

if __name__ == "__main__":
    main()
