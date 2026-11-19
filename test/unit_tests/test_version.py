"""The installed package and computed release versions."""

import shutil
import subprocess  # nosec B404 - runs Git and the release script in a disposable repo
import sys
from functools import partial
from pathlib import Path

import pytest
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


@pytest.fixture
def release_repo(tmp_path):
    git_path = shutil.which("git")
    assert git_path is not None

    def run(*args):
        # Only executes this test's Git commands and the checked-out release script.
        return subprocess.check_output(args, cwd=tmp_path, text=True).strip()  # nosec B603

    git = partial(run, git_path)
    release = partial(run, sys.executable, str(SCRIPT))
    git("init", "-q", "-b", "master")
    git("config", "user.name", "Release test")
    git("config", "user.email", "release@example.invalid")
    return git, release


def test_release_versions_follow_master_history(tmp_path, release_repo):
    git, release = release_repo
    version_file = tmp_path / "version.py"
    version_file.write_text('__version__ = "2.1.0dev0"\n')
    git("add", "version.py")
    git("commit", "-qm", "first")
    assert release("check-changes") == "skip=false"
    git("tag", "v1.9.0")
    git("tag", "v1.10.0")
    first = release("dev-build", "--init-file", version_file)
    assert first == release("dev-build", "--init-file", version_file)

    git("commit", "--allow-empty", "-qm", "squashed change")
    git("tag", "v2.0.0rc1")
    git("tag", "v9.9.9-test")
    assert release("check-changes") == "skip=false"
    assert git("rev-list", "--count", "v1.10.0..HEAD") == "1"
    second = release("dev-build", "--init-file", version_file)
    assert (
        Version("2.1.0.dev4294967295")
        < Version(first)
        < Version(second)
        < Version("2.1.0")
    )
    assert version_file.read_text() == f'__version__ = "{second}"\n'


def test_stable_release_ignores_only_its_empty_version_bump(tmp_path, release_repo):
    git, release = release_repo
    version = tmp_path / "braindecode/version.py"
    notes = tmp_path / "docs/whats_new.rst"
    version.parent.mkdir()
    notes.parent.mkdir()
    version.write_text('__version__ = "2.1.0dev0"\n')
    notes.write_text(
        ".. _current:\n\nCurrent 2.1.0 (GitHub)\n============\n\n- A change.\n"
    )
    git("add", ".")
    git("commit", "-qm", "tested master")
    git("checkout", "-q", "--detach")
    release("finalize")
    git("commit", "-qam", "release: v2.1.0")
    git("tag", "v2.1.0")
    git("checkout", "-q", "master")
    assert release("check-changes") == "skip=true"
    git("checkout", "-qb", "next-cycle")
    git(
        "restore",
        "--source",
        "v2.1.0",
        "--",
        "braindecode/version.py",
        "docs/whats_new.rst",
    )
    release("next-dev")
    git("commit", "-qam", "bump next development version")
    git("checkout", "-q", "master")
    git("merge", "--no-ff", "-qm", "merge the generated bump", "next-cycle")
    assert release("check-changes") == "skip=true"

    # A genuine note added to that same bump must still trigger publication.
    notes.write_text(notes.read_text() + "\n- Document a new dataset.\n")
    git("commit", "-qam", "add a release note", "--amend")
    assert release("check-changes") == "skip=false"
