"""The development version must sort after the last release."""

from packaging.version import Version

import braindecode


def test_development_version_sorts_after_the_last_release():
    # Downstream code gates workarounds on the version: NeuralBench keeps a LoRA
    # pre-hook for LaBraM's qkv "until braindecode > 1.8.1" (braindecode#1194).
    # With master still reporting 1.8.1 the gate cannot tell the two apart, and
    # the hook adds the LoRA delta a second time on master.
    assert Version(braindecode.__version__) > Version("1.8.1")
    assert Version(braindecode.__version__).is_prerelease
