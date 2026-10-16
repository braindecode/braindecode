# Authors: Anton Soloviev <anton@praviel.com>
# License: BSD (3-clause)

import copy
import json

import numpy as np
import pytest

from braindecode.preprocessing import Crop, Pick, Preprocessor


@pytest.mark.parametrize("preprocessor", [
    Preprocessor("crop", tmax=1, apply_on_array=False),
    Preprocessor(np.clip, a_min=-1, a_max=1),
    Crop(tmax=1),
    Pick(picks=[0, 1]),
])
def test_deserialize_keeps_reusable_configuration(preprocessor):
    config = json.loads(json.dumps(preprocessor.serialize()))
    expected = copy.deepcopy(config)

    first = Preprocessor.deserialize(config)

    assert config == expected
    assert json.loads(json.dumps(config)) == expected
    second = Preprocessor.deserialize(config)
    assert first == second == preprocessor


def test_failed_deserialize_keeps_configuration():
    config = {"__class_path__": "braindecode.preprocessing.preprocess.Missing"}
    with pytest.raises(AttributeError):
        Preprocessor.deserialize(config)
    assert config == {"__class_path__": "braindecode.preprocessing.preprocess.Missing"}
