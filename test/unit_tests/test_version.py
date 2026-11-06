"""The package version must sort after 1.8.1."""

from packaging.version import Version

import braindecode


def test_version_sorts_after_1_8_1():
    # Downstream code gates workarounds on the version: NeuralBench keeps a LoRA
    # pre-hook for LaBraM's qkv "until braindecode > 1.8.1" (braindecode#1194).
    # With master still reporting 1.8.1 the gate cannot tell the two apart, and
    # the hook adds the LoRA delta a second time on master. Released versions
    # (the tag commit of a release, and the sdist built from it) must pass too.
    assert Version(braindecode.__version__) > Version("1.8.1")
