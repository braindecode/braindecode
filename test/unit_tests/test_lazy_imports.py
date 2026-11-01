import importlib
import pkgutil
import subprocess  # nosec B404 - runs constant snippets with sys.executable
import sys

import braindecode
from braindecode import models

CODE = """
import sys
import braindecode
assert "braindecode.models.labram" not in sys.modules
import braindecode.models
assert not [m for m in sys.modules if m.startswith("braindecode.models.")]
from braindecode.models import Labram
assert Labram.__module__ == "braindecode.models.labram"
assert "braindecode.models.labram" in sys.modules
assert "braindecode.models.eegnet" not in sys.modules
assert "sklearn" not in sys.modules and "pandas" not in sys.modules
"""

# Every subpackage stays reachable as an attribute after a bare ``import braindecode``.
SUBMODULES_CODE = """
import importlib, pkgutil, sys
import braindecode
assert "braindecode.models" not in sys.modules
for m in pkgutil.iter_modules(braindecode.__path__):
    assert getattr(braindecode, m.name) is importlib.import_module("braindecode." + m.name), m.name
assert braindecode.models.EEGNet.__name__ == "EEGNet"
"""

# The registry is published in one step: no line of _init_models_dict runs
# while models_dict is partly filled (another thread could read it then).
ATOMIC_CODE = """
import sys
import braindecode.models.util as util
seen = []
def tracer(frame, event, arg):
    if frame.f_code is util._init_models_dict.__code__:
        if event == "line":
            seen.append(len(util._models_dict))
        return tracer
sys.settrace(tracer)
registry = util.models_dict
sys.settrace(None)
assert registry and set(seen) == {0}, sorted(set(seen))
"""


def _run(code):
    # Fresh interpreter: the test session has already imported every model.
    subprocess.run([sys.executable, "-c", code], check=True)  # nosec B603 - own code


def test_import_braindecode_does_not_load_models():
    _run(CODE)


def test_subpackages_load_on_attribute_access():
    _run(SUBMODULES_CODE)
    names = {m.name for m in pkgutil.iter_modules(braindecode.__path__)}
    assert names - {"version"} == braindecode._SUBMODULES


def test_models_dict_is_published_at_once():
    _run(ATOMIC_CODE)


def test_every_public_name_resolves_from_its_submodule():
    # __all__ and _LAZY are read from the TYPE_CHECKING imports of models/__init__.py.
    assert {"EEGNet", "Labram", "_init_models_dict"} <= set(models._LAZY)
    for name, module in models._LAZY.items():
        submodule = importlib.import_module(f"braindecode.models.{module}")
        assert getattr(models, name) is getattr(submodule, name), name
