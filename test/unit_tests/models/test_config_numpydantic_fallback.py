# Authors: Sarthak Tayal <sarthaktayal2@gmail.com>
import importlib
import sys
import unittest.mock

import pytest
from mne.utils import _soft_import

pydantic = _soft_import(name="pydantic", purpose="model config testing", strict=False)

if pydantic is False:
    pytest.skip("pydantic not installed, skipping", allow_module_level=True)


def test_config_import_without_numpydantic():
    # config module should load fine when numpydantic is missing
    saved = sys.modules.pop("numpydantic", None)
    try:
        with unittest.mock.patch.dict(sys.modules, {"numpydantic": None}):
            import braindecode.models.config as cfg_mod

            importlib.reload(cfg_mod)
            assert hasattr(cfg_mod, "ChsInfoType")
            assert "loc" in cfg_mod.ChsInfoType.__annotations__
    finally:
        if saved is not None:
            sys.modules["numpydantic"] = saved
