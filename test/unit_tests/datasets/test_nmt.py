# Authors: Mohammad Bayazi <mj.darvishi92@gmail.com>
#
# License: BSD-3
import platform
import pytest


from braindecode.datasets.nmt import _NMTMock

# Skip if OS is Windows
@pytest.mark.skipif(
    platform.system() == "Windows", reason="Not supported on Windows"
)  # TODO: Fix this
def test_nmt():
    nmt = _NMTMock(
        path="",
        n_jobs=1,  # required for test to work. mocking seems to fail otherwise
    )
    assert len(nmt.datasets) == 7
    assert nmt.description.shape == (7, 7)
    # assert nmt.description.version.to_list() == [
    #     'v2.0.0', 'v2.0.0', 'v2.0.0', 'v2.0.0', 'v2.0.0']
    assert nmt.description.pathological.to_list() == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert nmt.description.train.to_list() == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    x, y = nmt[0]
    assert x.shape == (21, 1)
    assert y is False
    x, y = nmt[-1]
    assert y is False

    nmt = _NMTMock(
        path="",
        target_name="age",
        n_jobs=1,
    )
    x, y = nmt[-1]
    assert y == 71
    for ds in nmt.datasets:
        ds.target_name = "gender"
    x, y = nmt[0]
    assert y == "M"
