"""
A smoke test for the wellshuffled package.

This script is intended to be run after the package has been installed.
It checks that the command-line tool is available and can be run.
"""

import subprocess
import sys


def main():
    """Run the smoke test."""
    try:
        # It's good practice to run the command with a version or help flag
        # to check that it's installed and executable.
        result = subprocess.run(
            ["wellshuffled", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Smoke test passed!")
        print(f"Output: {result.stdout}")
        sys.exit(0)
    except FileNotFoundError:
        print("Error: 'wellshuffled' command not found.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}.")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
