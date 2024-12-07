"""Main entry point for the Polaris package when run as a module.

This module enables running Polaris directly using 'python -m polaris'.
It provides a simple command router to different submodules.
"""

import asyncio
import sys

from . import cli


def main():
    """Main entry point for the package."""
    if len(sys.argv) < 2:
        print("Usage: python -m polaris <command> [args...]")
        print("\nAvailable commands:")
        print("  cli - Access the CLI interface")
        sys.exit(1)

    command = sys.argv.pop(1)  # Remove the command and shift remaining args

    if command == "cli":
        asyncio.run(cli.main())
    else:
        print(f"Unknown command: {command}")
        print("Available commands: cli")
        sys.exit(1)


if __name__ == "__main__":
    main()
