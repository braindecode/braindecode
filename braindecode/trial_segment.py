import logging
import numpy as np
import xarray as xr

log = logging.getLogger(__name__)


def create_target_series(cnt, marker_def, ival):
    """
    Compute one-hot encoded target series.
    Parameters
    ----------
    cnt : DateArray
    marker_def: OrderedDict (str -> list int)
        Dictionary mapping class names -> list of marker codes. Order of keys
         is used to determine indices of classes in one-hot encoded result. 
    ival : (number, number) tuple
        Segmentation interval for each trial.
    Returns
    -------
    2darray
        One-hot encoded target series.

    """
    assert 'fs' in cnt.attrs
    assert 'events' in cnt.attrs
    assert ival[0] < ival[1]
    sample_ival = cnt.attrs['fs'] * np.array(ival) / 1000.0
    sample_ival = np.int32(np.round(sample_ival))
    assert sample_ival[0] < sample_ival[1]


    class_names = marker_def.keys()
    targets = np.zeros((len(cnt.data), len(class_names)), dtype=np.int32)
    for i_sample, m in cnt.attrs['events']:
        for class_idx, classname in enumerate(class_names):
            if m in marker_def[classname]:
                start_index = i_sample + sample_ival[0]
                stop_index = i_sample + sample_ival[1]
                assert stop_index > start_index
                if start_index >= 0 and stop_index <= len(cnt.coords['time']):
                    targets[start_index:stop_index, class_idx] = 1
                elif start_index < 0:
                    log.warn("Ignoring trial, start index < 0: {:d}".format(
                        start_index))
                else:
                    assert stop_index > len(cnt.coords['time'])
                    log.warn("Ignoring trial, start index > n_samples:"
                        "{:d} > {:d} ".format(
                        stop_index, len(cnt.coords['time'])))
    return targets


def compute_trial_start_stop_samples(y, check_trial_lengths_equal=True,
                                    input_time_length=None):
    """Computes trial start and end samples (end is inclusive) from
    one-hot encoded y-matrix.
    Specify input time length to kick out trials that are too short after
    signal start.

    Parameters
    ----------
    y : 2darray
    check_trial_lengths_equal : bool
         (Default value = True)
    input_time_length : int, optional
         (Default value = None)

    Returns
    -------

    """
    #TODO: change to start stop
    trial_part = np.sum(y, 1) == 1
    boundaries = np.diff(trial_part.astype(np.int32))
    i_trial_starts = np.flatnonzero(boundaries == 1) + 1
    i_trial_stops = np.flatnonzero(boundaries == -1) + 1
    # it can happen that a trial is only partially there since the
    # cnt signal was split in the middle of a trial
    # for now just remove these
    # use that start marker should always be before or equal to end marker
    if i_trial_starts[0] >= i_trial_stops[0]:
        # cut out first trial which only has end marker
        i_trial_stops = i_trial_stops[1:]
    if i_trial_starts[-1] >= i_trial_stops[-1]:
        # cut out last trial which only has start marker
        i_trial_starts = i_trial_starts[:-1]

    assert (len(i_trial_starts) == len(i_trial_stops))
    assert (np.all(i_trial_starts < i_trial_stops))
    # possibly remove first trials if they are too early
    if input_time_length is not None:
        while i_trial_starts[0] < (input_time_length - 1):
            i_trial_starts = i_trial_starts[1:]
            i_trial_stops = i_trial_stops[1:]
    if check_trial_lengths_equal:
        # just checking that all trial lengths are equal
        all_trial_lens = np.array(i_trial_stops) - np.array(i_trial_starts)
        assert all(all_trial_lens == all_trial_lens[0]), (
            "All trial lengths should be equal...")
    return i_trial_starts, i_trial_stops


def segment_dat(cnt, marker_def, ival):
    """Convert a continuous data object to an epoched one.

    Given a continuous data object, a definition of classes, and an
    interval, this method looks for markers as defined in ``marker_def``
    and slices the dat according to the time interval given with
    ``ival`` along the ``timeaxis``. The returned ``dat`` object stores
    those slices and the class each slice belongs to.

    Epochs that are too close to the borders and thus too short are
    ignored.

    If the segmentation does not result in any epochs (i.e. the markers
    in ``marker_def`` could not be found in ``dat``, the resulting
    epo.data will be an empty array.

    This method is also suitable for **online processing**, please read
    the documentation for the ``newsamples`` parameter and have a look
    at the Examples below.

    Parameters
    ----------
    dat : Data
        the data object to be segmented
    marker_def : dict
        The keys are class names, the values are lists of markers
    ival : [int, int]
        The interval in milliseconds to cut around the markers. I.e. to
        get the interval starting with the marker plus the remaining
        100ms define the interval like [0, 100]. The start point is
        included, the endpoint is not (like: ``[start, end)``).  To get
        200ms before the marker until 100ms after the marker do:
        ``[-200, 100]`` Only negative or positive values are possible
        (i.e. ``[-500, -100]``)

    Returns
    -------
    epo : Data
        a copy of the resulting epoched data.

    Raises
    ------
    AssertionError
        * if ``dat`` has not ``.fs`` or ``.markers`` attribute or if
          ``ival[0] > ival[1]``.

    Examples
    --------

    Offline Experiment

    >>> # Define the markers belonging to class 1 and 2
    >>> md = {'class 1': ['S1', 'S2'],
    ...       'class 2': ['S3', 'S4']
    ...      }
    >>> # Epoch the data -500ms and +700ms around the markers defined in
    >>> # md
    >>> epo = segment_dat(cnt, md, [-500, 700])

    """

    assert 'fs' in cnt.attrs
    assert 'events' in cnt.attrs
    y = create_target_series(cnt, marker_def, ival=ival)
    # Create classes per trial
    # and sample inds per trial from the target series
    starts, stops = compute_trial_start_stop_samples(y)
    classes = [np.argmax(y[i_s]) for i_s in starts]
    sample_inds_per_trial = [list(range(i_start, i_stop))
                             for (i_start, i_stop) in zip(starts, stops)]

    if len(sample_inds_per_trial) == 0:
        assert "No epochs not tested yet"
        data = np.array([])
    else:
        timeaxis = list(cnt.dims).index('time')
        # np.take inserts a new dimension at `axis`...
        data = cnt.data.take(sample_inds_per_trial, axis=timeaxis)
        # we want that new dimension at axis 0 so we have to swap it.
        # before that we have to convert the negative axis indices to
        # their equivalent positive one, otherwise swapaxis will be one
        # off.
        data = data.swapaxes(0, timeaxis)
    time = np.linspace(ival[0], ival[1],
                       data.shape[1], # assume timeaxis now on 1
                       endpoint=False)
    epo = xr.DataArray(data,
                       coords={'trials': classes, 'time': time,
                               'channels': cnt.channels,},
                       dims=('trials', 'time','channels',))
    epo.attrs = cnt.attrs.copy()
    # remove events part
    epo.attrs.pop('events')
    return epo

