"""The package version, and the development versions ``release.yml`` publishes."""

from pathlib import Path
from runpy import run_path

from packaging.version import Version

import braindecode


def test_version_sorts_after_1_8_1():
    # Downstream code gates workarounds on the version: NeuralBench keeps a LoRA
    # pre-hook for LaBraM's qkv "until braindecode > 1.8.1" (braindecode#1194).
    # With master still reporting 1.8.1 the gate cannot tell the two apart, and
    # the hook adds the LoRA delta a second time on master. Released versions
    # (the tag commit of a release, and the sdist built from it) must pass too.
    assert Version(braindecode.__version__) > Version("1.8.1")


def test_dev_builds_follow_commit_order(tmp_path):
    # ``pip install --pre`` takes the highest version, so a build must outrank the
    # hash-numbered builds published earlier (below 2**32) and the builds of older
    # master commits, yet stay below the stable release.
    script = Path(__file__).resolve().parents[2] / "scripts" / "monthly_release.py"
    release = run_path(str(script))
    version_file = tmp_path / "version.py"
    version_file.write_text('__version__ = "2.1.0dev0"\n')

    def build(commits):
        args = ["dev-build", str(commits), "--init-file", str(version_file)]
        assert release["main"](args) == 0
        return Version(release["read_version"](version_file))

    older, newer = build(1367), build(1368)
    assert Version(f"2.1.0.dev{2**32 - 1}") < older < newer < Version("2.1.0")
