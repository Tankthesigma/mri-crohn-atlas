#!/usr/bin/env python3
"""
Fix and Reset Script for MRI-Crohn Atlas
Part 1: Fix Execution Errors & Set Key

This script:
1. Safely deletes the progress file using os.remove() (not bash rm)
2. Sets the OPENROUTER_API_KEY environment variable for the session
"""

import os
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("FIX AND RESET SCRIPT")
    print("=" * 60)

    # Define paths
    project_root = Path(__file__).parent.parent.parent
    calibration_dir = Path(__file__).parent

    # Progress file that may need to be deleted
    progress_file = calibration_dir / "parser_validation_progress.json"

    # Also check in parser_validation directory
    parser_validation_progress = project_root / "data" / "parser_validation" / "real_validation_progress.json"

    # Step 1: Delete progress files if they exist
    print("\n[Step 1] Cleaning up progress files...")

    files_to_delete = [
        progress_file,
        parser_validation_progress,
    ]

    for filepath in files_to_delete:
        if filepath.exists():
            try:
                os.remove(filepath)
                print(f"  ✓ Deleted: {filepath}")
            except Exception as e:
                print(f"  ✗ Failed to delete {filepath}: {e}")
        else:
            print(f"  - Not found (OK): {filepath}")

    # Step 2: Set the API key as environment variable
    print("\n[Step 2] Setting OPENROUTER_API_KEY environment variable...")

    # The new API key provided
    api_key = "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"

    # Set it in the current process environment
    os.environ["OPENROUTER_API_KEY"] = api_key

    # Verify it was set
    if os.environ.get("OPENROUTER_API_KEY") == api_key:
        print(f"  ✓ API key set successfully")
        print(f"  ✓ Key prefix: {api_key[:15]}...")
    else:
        print(f"  ✗ Failed to set API key")
        return False

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nThe OPENROUTER_API_KEY is now set for this Python session.")
    print("You can now run the validation script from this same process.")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
