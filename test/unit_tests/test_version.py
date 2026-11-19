"""The installed package and computed release versions."""

import subprocess
import sys
from pathlib import Path

from packaging.version import Version

import braindecode


def test_version_sorts_after_1_8_1():
    # Downstream code gates workarounds on the version: NeuralBench keeps a LoRA
    # pre-hook for LaBraM's qkv "until braindecode > 1.8.1" (braindecode#1194).
    # With master still reporting 1.8.1 the gate cannot tell the two apart, and
    # the hook adds the LoRA delta a second time on master. Released versions
    # (the tag commit of a release, and the sdist built from it) must pass too.
    assert Version(braindecode.__version__) > Version("1.8.1")


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "monthly_release.py"


def test_release_versions_follow_master_history(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    def release(*args):
        return subprocess.check_output(
            [sys.executable, str(SCRIPT), *args], cwd=tmp_path, text=True
        ).strip()

    git("init", "-q")
    git("config", "user.name", "Release test")
    git("config", "user.email", "release@example.invalid")
    version_file = tmp_path / "version.py"
    version_file.write_text('__version__ = "2.1.0dev0"\n')
    git("add", "version.py")
    git("commit", "-qm", "first")
    assert release("latest-tag") == ""
    git("tag", "v1.9.0")
    git("tag", "v1.10.0")
    first = release("dev-build", "--init-file", version_file)
    assert first == release("dev-build", "--init-file", version_file)

    git("commit", "--allow-empty", "-qm", "squashed change")
    git("tag", "v2.0.0rc1")
    git("tag", "v9.9.9-test")
    assert release("latest-tag") == "v1.10.0"
    assert git("rev-list", "--count", "v1.10.0..HEAD") == "1"
    second = release("dev-build", "--init-file", version_file)
    assert (
        Version("2.1.0.dev4294967295")
        < Version(first)
        < Version(second)
        < Version("2.1.0")
    )
    assert version_file.read_text() == f'__version__ = "{second}"\n'
