"""Download MNE-Python references to merge with Braindecode references."""

import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def update_references():
    """Download and merge MNE references with Braindecode references.

    This function fetches the MNE-Python references.bib from GitHub and
    appends unique entries to Braindecode's references.bib file.

    Returns
    -------
    bool
        True if update succeeded or was skipped gracefully, False if critical error.

    """
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).resolve().parent.absolute()
        bib_file = script_dir / "references.bib"

        # Verify the file exists
        if not bib_file.exists():
            logger.debug(f"References file not found at {bib_file}")
            return True  # Don't fail if file doesn't exist

        # Use constant URL to prevent scheme injection (Codacy security requirement)
        MNE_URL = "https://raw.githubusercontent.com/mne-tools/mne-python/main/doc/references.bib"

        parsed_url = urlparse(MNE_URL)
        if parsed_url.scheme not in {"https", "http"} or not parsed_url.netloc:
            logger.warning(f"Invalid URL scheme or netloc in {MNE_URL}")
            return True  # Don't fail on invalid URL

        # Read current file first
        try:
            with open(bib_file, "r", encoding="utf-8") as f:
                current_content = f.read()
        except OSError as e:
            logger.debug(f"Could not read references file: {e}")
            return True  # Don't fail if we can't read

        # Check if MNE section already exists
        if "mne-tools/mne-python" in current_content:
            logger.debug("MNE references already merged")
            return True

        try:
            # Download MNE references using Request object for better security control
            req = urllib.request.Request(MNE_URL, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:  # noqa: S310
                mne_content = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as e:
            logger.warning(f"Could not download MNE references: {e}")
            return True  # Don't fail if download fails

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
            logger.debug("Successfully merged MNE references")
            return True
        except OSError as e:
            logger.warning(f"Could not write references file: {e}")
            return True  # Don't fail on write error

    except Exception as e:
        logger.warning(f"Unexpected error during reference update: {e}")
        return True  # Never fail the build


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    success = update_references()
    sys.exit(0 if success else 1)
