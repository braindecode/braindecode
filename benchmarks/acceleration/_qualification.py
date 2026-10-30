"""Explicit boundary between historical diagnostics and new qualification.

These recipes preserve old audited records; they do not attest launch topology,
source integrity or policy receipts. A schema-shaped record is not an attestation.
"""
import sys


def historical_only(record, reason):
    """Label a diagnostic record so status=ok cannot imply new qualification."""
    boundary = {
        "status": "historical_only",
        "new_claim_eligible": False,
        "reason": reason,
        "required_for_new_claim": [
            "retained launch and resource/topology receipt",
            "clean source commit and independently checked source hashes",
            "explicit environment, scheduler and measurement policy receipt",
            "finite raw samples, stability and complete paired-run checks",
        ],
    }
    if "schema_version" in record:
        record["config"]["parameters"]["qualification"] = boundary
    else:
        record["qualification"] = boundary
    return record


def require_linux_rss():
    """Reject non-Linux ru_maxrss units before a Linux-only RSS measurement."""
    if sys.platform != "linux":
        raise RuntimeError("This historical RSS recipe requires Linux (ru_maxrss KiB).")
