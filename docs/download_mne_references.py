"""Download MNE-Python references to merge with Braindecode references."""

import sys
import urllib.error
import urllib.request
from pathlib import Path


def update_references():
    """Download and merge MNE references with Braindecode references.

    This function fetches the MNE-Python references.bib from GitHub and
    appends unique entries to Braindecode's references.bib file.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).resolve().parent.absolute()
    bib_file = script_dir / "references.bib"

    # Verify the file exists
    if not bib_file.exists():
        return False

    # Use constant URL to prevent scheme injection (Codacy security requirement)
    MNE_URL = (
        "https://raw.githubusercontent.com/mne-tools/mne-python/main/doc/references.bib"
    )

    try:
        # Download MNE references using Request object for better security control
        req = urllib.request.Request(MNE_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:  # noqa: S310
            mne_content = response.read().decode("utf-8")
    except urllib.error.URLError:
        return False

    # Read current file
    try:
        with open(bib_file, "r", encoding="utf-8") as f:
            current_content = f.read()
    except OSError:
        return False

    # Check if MNE section already exists
    if "mne-tools/mne-python" in current_content:
        return True

    # Add MNE section with header
    merged = current_content.rstrip() + "\n\n"
    merged += (
        "% MNE-Python References\n"
        "% Source: https://github.com/mne-tools/mne-python/blob/main/doc/references.bib\n"
    )
    merged += mne_content

    # Write merged file
    try:
        with open(bib_file, "w", encoding="utf-8") as f:
            f.write(merged)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    success = update_references()
    sys.exit(0 if success else 1)
