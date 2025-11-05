"""Tests for channel utilities in braindecode.datautil.channel_utils."""

import pytest

from braindecode.datautil.channel_utils import (
    division_channels_idx,
    match_hemisphere_chans,
)


@pytest.fixture()
def standard_channels():
    return ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']


@pytest.fixture()
def full_22_channels():
    return [
        'FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'O1', 'O2'
    ]


def test_division_channels_idx(standard_channels):
    """Test channel division into left, right, and middle."""
    left, right, middle = division_channels_idx(standard_channels)

    assert left == ['FP1', 'F3', 'C3', 'P3', 'O1']
    assert right == ['FP2', 'F4', 'C4', 'P4', 'O2']
    assert middle == ['Fz', 'Cz', 'Pz']

    # Test with only odd channels
    left, right, middle = division_channels_idx(['C3', 'F3', 'P3'])
    assert left == ['C3', 'F3', 'P3']
    assert right == []
    assert middle == []

    # Test with empty list
    left, right, middle = division_channels_idx([])
    assert left == []
    assert right == []
    assert middle == []


def test_match_hemisphere_chans():
    """Test matching of left and right hemisphere channels."""
    left = ['C3', 'F3', 'P3']
    right = ['C4', 'F4', 'P4']
    result = match_hemisphere_chans(left, right)
    assert result == [('C3', 'C4'), ('F3', 'F4'), ('P3', 'P4')]

    # Test with out-of-order right channels
    right = ['P4', 'C4', 'F4']
    result = match_hemisphere_chans(left, right)
    assert result == [('C3', 'C4'), ('F3', 'F4'), ('P3', 'P4')]


def test_match_hemisphere_chans_errors():
    """Test error cases for match_hemisphere_chans."""
    # Unequal length
    with pytest.raises(ValueError, match="Left and right channels do not match"):
        match_hemisphere_chans(['C3', 'F3'], ['C4'])

    # Channel without number
    with pytest.raises(ValueError, match="does not contain a number"):
        match_hemisphere_chans(['Cz'], ['C4'])

    # No matching right channel
    with pytest.raises(ValueError, match="Found no right hemisphere matching channel"):
        match_hemisphere_chans(['C3', 'F3'], ['C4', 'P4'])


def test_division_and_matching_workflow(standard_channels):
    """Test typical workflow: divide channels, then match hemispheres."""
    left, right, middle = division_channels_idx(standard_channels)

    assert left == ['FP1', 'F3', 'C3', 'P3', 'O1']
    assert right == ['FP2', 'F4', 'C4', 'P4', 'O2']
    assert middle == ['Fz', 'Cz', 'Pz']

    matched = match_hemisphere_chans(left, right)
    assert matched == [('FP1', 'FP2'), ('F3', 'F4'), ('C3', 'C4'), ('P3', 'P4'), ('O1', 'O2')]


def test_full_channel_setup(full_22_channels):
    """Test with a typical 22-channel EEG setup."""
    left, right, middle = division_channels_idx(full_22_channels)

    assert len(left) == 10
    assert len(right) == 10
    assert len(middle) == 2

    matched = match_hemisphere_chans(left, right)
    assert len(matched) == 10
