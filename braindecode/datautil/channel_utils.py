"""
Utilities for EEG channel manipulation and selection.

This module provides functions for dividing and matching EEG channels,
particularly for hemisphere-aware processing.
"""

import re
from re import search


def match_hemisphere_chans(left_chs, right_chs):
    """
    Match channels of the left and right hemispheres based on their names.

    This function pairs channels from the left and right hemispheres by matching
    their numeric identifiers. For a left channel with number N, it finds the
    corresponding right channel with number N+1.

    Parameters
    ----------
    left_chs : list of str
        A list of channel names from the left hemisphere.
    right_chs : list of str
        A list of channel names from the right hemisphere.

    Returns
    -------
    list of tuples
        List of tuples with matched channel names from the left and right hemispheres.
        Each tuple contains (left_channel, right_channel).

    Raises
    ------
    ValueError
        If the left and right channels do not match in length.
    ValueError
        If a channel name does not contain a number.
    ValueError
        If no matching right hemisphere channel is found for a left channel.

    Examples
    --------
    >>> left = ['C3', 'F3']
    >>> right = ['C4', 'F4']
    >>> match_hemisphere_chans(left, right)
    [('C3', 'C4'), ('F3', 'F4')]
    """
    if len(left_chs) != len(right_chs):
        raise ValueError("Left and right channels do not match.")
    right_chs = list(right_chs)
    regexp = r"\d+"
    out = []
    for left in left_chs:
        match = re.search(regexp, left)
        if match is None:
            raise ValueError(f"Channel '{left}' does not contain a number.")
        chan_idx = 1 + int(match.group())
        target_r = re.sub(regexp, str(chan_idx), left)
        for right in right_chs:
            if right == target_r:
                out.append((left, right))
                right_chs.remove(right)
                break
        else:
            raise ValueError(
                f"Found no right hemisphere matching channel for '{left}'."
            )
    return out


def division_channels_idx(ch_names):
    """
    Divide EEG channel names into left, right, and middle based on numbering.

    This function categorizes channels by their numeric suffix:
    - Odd-numbered channels → left hemisphere
    - Even-numbered channels → right hemisphere
    - Channels without numbers → middle/midline

    Parameters
    ----------
    ch_names : list of str
        A list of EEG channel names to be divided based on their numbering.

    Returns
    -------
    tuple of lists
        Three lists containing the channel names:
        - left: Odd-numbered channels (e.g., C3, F3, P3)
        - right: Even-numbered channels (e.g., C4, F4, P4)
        - middle: Channels without numbers (e.g., Cz, Fz, Pz)

    Notes
    -----
    The function identifies channel numbers by searching for numeric characters
    in the channel names. Standard 10-20 system EEG channel naming conventions
    use odd numbers for left hemisphere and even numbers for right hemisphere.

    Examples
    --------
    >>> channels = ['FP1', 'FP2', 'O1', 'O2', 'FZ']
    >>> division_channels_idx(channels)
    (['FP1', 'O1'], ['FP2', 'O2'], ['FZ'])
    """
    left, right, middle = [], [], []
    for ch in ch_names:
        number = search(r"\d+", ch)
        if number is not None:
            (left if int(number[0]) % 2 else right).append(ch)
        else:
            middle.append(ch)

    return left, right, middle
