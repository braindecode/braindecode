import importlib
import pkgutil
import py_compile
import subprocess  # nosec B404 - runs constant snippets with sys.executable
import sys
from pathlib import Path
from zipfile import ZipFile

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


def test_model_imports_and_summary_from_zip(tmp_path):
    package = Path(braindecode.__file__).parent
    archive = tmp_path / "braindecode.zip"
    registry = package / "models" / "__init__.py"
    bytecode = tmp_path / "__init__.pyc"
    py_compile.compile(str(registry), cfile=str(bytecode), doraise=True)
    with ZipFile(archive, "w") as zipped:
        for path in package.rglob("*"):
            if path == registry:
                zipped.write(bytecode, "braindecode/models/__init__.pyc")
            elif path.is_file() and path.suffix in {".py", ".pyi", ".csv"}:
                zipped.write(path, path.relative_to(package.parent))

    code = """
import sys
sys.path.insert(0, sys.argv[1])
from braindecode import models
assert sys.argv[1] in models.__file__ and models.__file__.endswith(".pyc")
assert not any(name.startswith('braindecode.models.') for name in sys.modules)
from braindecode.models import EEGNet
from braindecode.models.util import get_summary_table
assert EEGNet is models.eegnet.EEGNet
assert 'EEGNet' in get_summary_table().index
assert 'braindecode.models.labram' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code, str(archive)], check=True)  # nosec B603


def test_subpackages_load_on_attribute_access():
    _run(SUBMODULES_CODE)
    names = {m.name for m in pkgutil.iter_modules(braindecode.__path__)}
    assert names - {"version"} == braindecode._SUBMODULES


def test_models_dict_is_published_at_once():
    _run(ATOMIC_CODE)


def test_every_public_name_resolves_from_its_submodule():
    # The typed exports and the runtime registry name the same classes.
    assert {"EEGNet", "Labram", "_init_models_dict"} <= set(models._LAZY)
    for name, module in models._LAZY.items():
        submodule = importlib.import_module(f"braindecode.models.{module}")
        assert getattr(models, name) is getattr(submodule, name), name


# wfdb is only needed to read PhysioNet Challenge 2018 files; with pandas >= 3.1
# its import emits a DeprecationWarning, which ``-W error`` turned into an error.
WFDB_CODE = """
import sys
import braindecode.datasets
assert "wfdb" not in sys.modules
"""


def test_import_datasets_does_not_load_wfdb():
    _run(WFDB_CODE)
