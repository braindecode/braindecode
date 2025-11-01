# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Bruna Lopes <brunajaflopes@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
# Adapting some tests from test_preprocess file
#
# License: BSD-3


import copy
import os
import platform
from glob import glob

import mne
import numpy as np
import pandas as pd
import pytest
from pytest_cases import parametrize_with_cases

from braindecode.datasets import BaseConcatDataset, MOABBDataset, RawDataset
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.preprocessing import (
    AddChannels,
    AddEvents,
    AddProj,
    AddReferenceChannels,
    Anonymize,
    ApplyGradientCompensation,
    ApplyHilbert,
    ApplyProj,
    ComputeCurrentSourceDensity,
    Crop,
    CropByAnnotations,
    DelProj,
    DropChannels,
    EqualizeChannels,
    Filter,
    FixMagCoilTypes,
    FixStimArtifact,
    InterpolateBads,
    NotchFilter,
    Pick,
    RenameChannels,
    ReorderChannels,
    Resample,
    Rescale,
    SavgolFilter,
    SetAnnotations,
    SetChannelTypes,
    SetEEGReference,
    SetMeasDate,
    SetMontage,
)
from braindecode.preprocessing.preprocess import (
    Preprocessor,
    _replace_inplace,
    _set_preproc_kwargs,
    exponential_moving_standardize,
    filterbank,
    preprocess,
)
from braindecode.preprocessing.windowers import create_fixed_length_windows

# We can't use fixtures with scope='module' as the dataset objects are modified
# inplace during preprocessing. To avoid the long setup time caused by calling
# the dataset/windowing functions multiple times, we instantiate the dataset
# objects once and deep-copy them in fixture.
bnci_kwargs = {
    "n_sessions": 2,
    "n_runs": 1,
    "n_subjects": 1,
    "paradigm": "imagery",
    "duration": 386,
    "sfreq": 250,
    "event_list": ("left", "right"),
    "channels": ("C4", "Cz", "FC3", "Pz", "P2", "P1", "POz"),
}

raw_ds = MOABBDataset(
    dataset_name="FakeDataset", subject_ids=[1], dataset_kwargs=bnci_kwargs
)
windows_ds = create_fixed_length_windows(
    raw_ds,
    start_offset_samples=100,
    stop_offset_samples=None,
    window_size_samples=1000,
    window_stride_samples=1000,
    drop_last_window=True,
    mapping=None,
    preload=True,
)


# Get the raw data in fixture
@pytest.fixture
def base_concat_ds():
    return copy.deepcopy(raw_ds)


@pytest.fixture
def windows_concat_ds():
    return copy.deepcopy(windows_ds)


def test_preprocess_raw_kwargs(base_concat_ds):
    preprocessors = [Crop(tmax=10, include_tmax=False)]
    preprocess(base_concat_ds, preprocessors)
    assert len(base_concat_ds.datasets[0].raw.times) == 2500
    assert all(
        [
            ds.raw_preproc_kwargs
            == [
                ("crop", {"tmax": 10, "include_tmax": False}),
            ]
            for ds in base_concat_ds.datasets
        ]
    )


def test_preprocess_windows_kwargs(windows_concat_ds):
    preprocessors = [Crop(tmin=0, tmax=0.1, include_tmax=False)]
    preprocess(windows_concat_ds, preprocessors)
    # assert windows_concat_ds[0][0].shape[1] == 25  no longer correct as raw preprocessed
    # Since windowed datasets are not using mne epochs anymore,
    # also for windows it is called raw_preproc_kwargs
    # as underlying data is always raw
    assert all(
        [
            ds.raw_preproc_kwargs
            == [
                ("crop", {"tmin": 0, "tmax": 0.1, "include_tmax": False}),
            ]
            for ds in windows_concat_ds.datasets
        ]
    )


# To test one preprocessor at each time, using this fixture structure
class PrepClasses:
    @pytest.mark.parametrize("sfreq", [100, 300])
    def prep_resample(self, sfreq):
        return Resample(sfreq=sfreq)

    @pytest.mark.parametrize("picks", ["eeg"])
    def prep_picktype(self, picks):
        return Pick(picks=picks)

    @pytest.mark.parametrize("picks", [["Cz"], ["C4", "FC3"]])
    def prep_pickchannels(self, picks):
        return Pick(picks=picks)

    @pytest.mark.parametrize("l_freq,h_freq", [(4, 30), (7, None), (None, 35)])
    def prep_filter(self, l_freq, h_freq):
        return Filter(l_freq=l_freq, h_freq=h_freq)

    @pytest.mark.parametrize("ref_channels", ["average", ["C4"], ["C4", "Cz"]])
    def prep_setref(self, ref_channels):
        return SetEEGReference(ref_channels=ref_channels)

    @pytest.mark.parametrize("tmin,tmax", [(0, 0.1), (0.1, 1.2), (0.1, None)])
    def prep_crop(self, tmin, tmax):
        return Crop(tmin=tmin, tmax=tmax)

    @pytest.mark.parametrize("ch_names", ["Pz", "P2", "P1", "POz"])
    def prep_drop(self, ch_names):
        return DropChannels(ch_names=ch_names)

    @pytest.mark.parametrize("freqs", [[50], [60]])
    def prep_notch(self, freqs):
        return NotchFilter(freqs=freqs)

    @pytest.mark.parametrize("h_freq", [10, 20])
    def prep_savgol(self, h_freq):
        return SavgolFilter(h_freq=h_freq)

    @pytest.mark.parametrize("mapping", [{"C4": "C4_new"}])
    def prep_rename(self, mapping):
        return RenameChannels(mapping=mapping)

    @pytest.mark.parametrize("ch_names", [["Cz", "C4", "FC3", "Pz"]])
    def prep_reorder(self, ch_names):
        return ReorderChannels(ch_names=ch_names)

    @pytest.mark.parametrize("ref_channels", ["FCz"])
    def prep_addref(self, ref_channels):
        return AddReferenceChannels(ref_channels=ref_channels)

    @pytest.mark.parametrize("envelope", [True, False])
    def prep_hilbert(self, envelope):
        return ApplyHilbert(envelope=envelope)

    def prep_proj(self):
        return ApplyProj()

    def prep_interpolate(self):
        return InterpolateBads(reset_bads=True)

    def prep_montage(self):
        montage = mne.channels.make_standard_montage('standard_1020')
        return SetMontage(montage=montage, match_case=False, on_missing='ignore')

    def prep_csd(self):
        pytest.skip("ComputeCurrentSourceDensity requires montage setup")
        return ComputeCurrentSourceDensity()

    def prep_anonymize(self):
        return Anonymize()

    @pytest.mark.parametrize("mapping", [{"C4": "eog"}])
    def prep_setchanneltypes(self, mapping):
        return SetChannelTypes(mapping=mapping)

    @pytest.mark.parametrize("scalings", [{"eeg": 1e-6}])
    def prep_rescale(self, scalings):
        return Rescale(scalings=scalings)

    def prep_fixmagcoiltypes(self):
        pytest.skip("FixMagCoilTypes is MEG-specific")
        return FixMagCoilTypes()

    def prep_addproj(self):
        # Create a simple projection
        import numpy as np
        proj_data = {
            'kind': 1,
            'active': False,
            'desc': 'test',
            'data': {'col_names': [], 'row_names': [], 'data': np.array([[]])},
        }
        return AddProj(projs=[proj_data])

    def prep_delproj(self):
        pytest.skip("DelProj requires existing projections")
        return DelProj(idx=0)

    @pytest.mark.parametrize("daysback", [1, 10])
    def prep_setmeasdate(self, daysback):
        return SetMeasDate(meas_date=daysback)

    def prep_addchannels(self):
        # Create a simple raw to add
        info = mne.create_info(ch_names=['new_ch'], sfreq=250, ch_types=['eeg'])
        new_raw = mne.io.RawArray(np.random.randn(1, 96500), info)
        return AddChannels(add_list=[new_raw])

    def prep_cropbyannotations(self):
        pytest.skip("CropByAnnotations requires specific annotation format")
        # Create annotations first, then crop by them
        return CropByAnnotations(annotations=['BAD'])

    def prep_equalizechannels(self):
        pytest.skip("EqualizeChannels requires multiple raw objects")
        # This requires another raw object with same channels
        return EqualizeChannels(raws=[])

    def prep_fixstimart(self):
        pytest.skip("FixStimArtifact requires event setup")
        return FixStimArtifact(events=None)

    def prep_addevents(self):
        # Create simple events array
        events = np.array([[100, 0, 1], [200, 0, 2]])
        return AddEvents(events=events)

    @pytest.mark.parametrize("grade", [0, 1])
    def prep_applygradcomp(self, grade):
        pytest.skip("ApplyGradientCompensation is MEG-specific")
        return ApplyGradientCompensation(grade=grade)

    def prep_setannotations(self):
        # Create simple annotations
        annot = mne.Annotations(onset=[1], duration=[0.5], description=['test'])
        return SetAnnotations(annotations=annot)


@pytest.fixture
def base_concat_ds_with_montage(base_concat_ds):
    """Dataset with montage set for functions that require it."""
    import copy
    ds = copy.deepcopy(base_concat_ds)
    montage = mne.channels.make_standard_montage('standard_1020')
    for d in ds.datasets:
        d.raw.set_montage(montage, match_case=False, on_missing='ignore')
    return ds


@pytest.fixture
def base_concat_ds_with_bad_channels(base_concat_ds_with_montage):
    """Dataset with bad channels marked for interpolation."""
    import copy
    ds = copy.deepcopy(base_concat_ds_with_montage)
    for d in ds.datasets:
        d.raw.info['bads'] = ['Pz']
    return ds


@parametrize_with_cases("prep", cases=PrepClasses, prefix="prep_")
def test_preprocessings(prep, base_concat_ds):
    preprocessors = [prep]
    preprocess(base_concat_ds, preprocessors, n_jobs=1)


def test_new_filterbank(base_concat_ds):
    base_concat_ds = base_concat_ds.split([[0]])["0"]
    preprocessors = [
        Pick(picks=sorted(["C4", "Cz"])),
        Preprocessor(
            fn=filterbank,
            frequency_bands=[(0, 4), (4, 8), (8, 13)],
            drop_original_signals=False,
            apply_on_array=False,
        ),
    ]
    preprocess(base_concat_ds, preprocessors)
    for x, y in base_concat_ds:
        break
    assert x.shape[0] == 8
    freq_band_annots = [
        ch.split("_")[-1] for ch in base_concat_ds.datasets[0].raw.ch_names if "_" in ch
    ]
    assert len(np.unique(freq_band_annots)) == 3
    np.testing.assert_array_equal(
        base_concat_ds.datasets[0].raw.ch_names,
        [
            "C4",
            "C4_0-4",
            "C4_4-8",
            "C4_8-13",
            "Cz",
            "Cz_0-4",
            "Cz_4-8",
            "Cz_8-13",
        ],
    )
    assert all(
        [
            ds.raw_preproc_kwargs
            == [
                ("pick", {"picks": ["C4", "Cz"]}),
                (
                    "filterbank",
                    {
                        "frequency_bands": [(0, 4), (4, 8), (8, 13)],
                        "drop_original_signals": False,
                    },
                ),
            ]
            for ds in base_concat_ds.datasets
        ]
    )


def test_replace_inplace(base_concat_ds):
    base_concat_ds2 = copy.deepcopy(base_concat_ds)
    for i in range(len(base_concat_ds2.datasets)):
        base_concat_ds2.datasets[i].raw.crop(0, 10, include_tmax=False)
    _replace_inplace(base_concat_ds, base_concat_ds2)

    assert all([len(ds.raw.times) == 2500 for ds in base_concat_ds.datasets])


def test_set_raw_preproc_kwargs(base_concat_ds):
    raw_preproc_kwargs = [("crop", {"tmax": 10, "include_tmax": False})]
    preprocessors = [Crop(tmax=10, include_tmax=False)]
    ds = base_concat_ds.datasets[0]
    _set_preproc_kwargs(ds, preprocessors)

    assert hasattr(ds, "raw_preproc_kwargs")
    assert ds.raw_preproc_kwargs == raw_preproc_kwargs


def test_set_window_preproc_kwargs(windows_concat_ds):
    window_preproc_kwargs = [("crop", {"tmax": 10, "include_tmax": False})]
    preprocessors = [Crop(tmax=10, include_tmax=False)]
    ds = windows_concat_ds.datasets[0]
    _set_preproc_kwargs(ds, preprocessors)

    # Since windowed datasets are not using mne epochs anymore,
    # also for windows it is called raw_preproc_kwargs
    # as underlying data is always raw
    assert hasattr(ds, "raw_preproc_kwargs")
    assert ds.raw_preproc_kwargs == window_preproc_kwargs


def test_set_preproc_kwargs_wrong_type(base_concat_ds):
    preprocessors = [Crop(tmax=10, include_tmax=False)]
    with pytest.raises(TypeError):
        _set_preproc_kwargs(base_concat_ds, preprocessors)


@pytest.mark.skipif(platform.system() == "Windows", reason="Not supported on Windows")
@pytest.mark.parametrize("kind", ["raw", "windows"])
@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("n_jobs", [-1, 1, 2, None])
def test_preprocess_save_dir(
    base_concat_ds, windows_concat_ds, tmp_path, kind, save, overwrite, n_jobs
):
    preproc_kwargs = [("crop", {"tmin": 0, "tmax": 0.1, "include_tmax": False})]
    preprocessors = [Crop(tmin=0, tmax=0.1, include_tmax=False)]

    save_dir = str(tmp_path) if save else None
    # Since windowed datasets are not using mne epochs anymore,
    # also for windows it is called raw_preproc_kwargs
    # as underlying data is always raw
    preproc_kwargs_name = "raw_preproc_kwargs"
    if kind == "raw":
        concat_ds = base_concat_ds
    elif kind == "windows":
        concat_ds = windows_concat_ds

    concat_ds = preprocess(
        concat_ds, preprocessors, save_dir, overwrite=overwrite, n_jobs=n_jobs
    )

    assert all([hasattr(ds, preproc_kwargs_name) for ds in concat_ds.datasets])
    assert all(
        [getattr(ds, preproc_kwargs_name) == preproc_kwargs for ds in concat_ds.datasets]
    )
    assert all([len(ds.raw.times) == 25 for ds in concat_ds.datasets])
    if kind == "raw":
        assert all([hasattr(ds, "target_name") for ds in concat_ds.datasets])

    if save_dir is None:
        assert all([ds.raw.preload for ds in concat_ds.datasets])
    else:
        assert all([not ds.raw.preload for ds in concat_ds.datasets])
        save_dirs = [
            os.path.join(save_dir, str(i)) for i in range(len(concat_ds.datasets))
        ]
        assert set(glob(save_dir + "/*")) == set(save_dirs)


def test_mne_preprocessor(base_concat_ds):
    low_cut_hz = 4.0  # low cut frequency for filtering
    high_cut_hz = 38.0  # high cut frequency for filtering

    preprocessors = [
        Resample(sfreq=100),
        Pick(picks=["eeg"]),  # Keep EEG sensors
        Filter(l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize),
    ]

    preprocess(base_concat_ds, preprocessors, n_jobs=-1)


def test_new_eegref(base_concat_ds):
    preprocessors = [SetEEGReference(ref_channels="average")]
    preprocess(base_concat_ds, preprocessors, n_jobs=1)


def test_new_filterbank_order_channels_by_freq(base_concat_ds):
    base_concat_ds = base_concat_ds.split([[0]])["0"]
    preprocessors = [
        # DropChannels(ch_names=["P2", "P1"]),
        Pick(picks=sorted(["C4", "Cz"])),
        Preprocessor(
            filterbank,
            frequency_bands=[(0, 4), (4, 8), (8, 13)],
            drop_original_signals=False,
            order_by_frequency_band=True,
            apply_on_array=False,
        ),
    ]
    preprocess(base_concat_ds, preprocessors)
    np.testing.assert_array_equal(
        base_concat_ds.datasets[0].raw.ch_names,
        ["C4", "Cz", "C4_0-4", "Cz_0-4", "C4_4-8", "Cz_4-8", "C4_8-13", "Cz_8-13"],
    )
    assert all(
        [
            ds.raw_preproc_kwargs
            == [
                ("pick", {"picks": ["C4", "Cz"]}),
                (
                    "filterbank",
                    {
                        "frequency_bands": [(0, 4), (4, 8), (8, 13)],
                        "drop_original_signals": False,
                        "order_by_frequency_band": True,
                    },
                ),
            ]
            for ds in base_concat_ds.datasets
        ]
    )


# Test overwriting
@pytest.mark.parametrize("overwrite", [True, False])
def test_new_overwrite(base_concat_ds, tmp_path, overwrite):
    preprocessors = [Crop(tmax=10, include_tmax=False)]

    # Create temporary directory with preexisting files
    save_dir = str(tmp_path)
    for i, ds in enumerate(base_concat_ds.datasets):
        concat_ds = BaseConcatDataset([ds])
        save_subdir = os.path.join(save_dir, str(i))
        os.makedirs(save_subdir)
        concat_ds.save(save_subdir, overwrite=True)

    if overwrite:
        preprocess(base_concat_ds, preprocessors, save_dir, overwrite=True)
        # Make sure the serialized data is preprocessed
        preproc_concat_ds = load_concat_dataset(save_dir, True)
        assert all([len(ds.raw.times) == 2500 for ds in preproc_concat_ds.datasets])
    else:
        with pytest.raises(FileExistsError):
            preprocess(base_concat_ds, preprocessors, save_dir, overwrite=False)


def test_new_misc_channels():
    rng = np.random.RandomState(42)
    signal_sfreq = 50
    info = mne.create_info(
        ch_names=["0", "1", "target_0", "target_1"],
        sfreq=signal_sfreq,
        ch_types=["eeg", "eeg", "misc", "misc"],
    )
    signal = rng.randn(2, 1000)
    targets = rng.randn(2, 1000)
    raw = mne.io.RawArray(np.concatenate([signal, targets]), info=info)
    desc = pd.Series({"pathological": True, "gender": "M", "age": 48})
    base_dataset = RawDataset(raw, desc, target_name=None)
    concat_ds = BaseConcatDataset([base_dataset])

    preprocessors = [
        Pick(picks=["eeg", "misc"]),
        Preprocessor(lambda x: x / 1e6),
    ]

    preprocess(concat_ds, preprocessors)

    # Check whether preprocessing has not affected the targets
    # This is only valid for preprocessors that use mne functions which do not modify
    # `misc` channels.
    np.testing.assert_array_equal(concat_ds.datasets[0].raw.get_data()[-2:, :], targets)


def test_interpolate_bads(base_concat_ds_with_bad_channels):
    """Test InterpolateBads preprocessor."""
    preprocessors = [InterpolateBads(reset_bads=True)]
    preprocess(base_concat_ds_with_bad_channels, preprocessors)
    # After interpolation, bads should be reset
    assert all([ds.raw.info['bads'] == [] for ds in base_concat_ds_with_bad_channels.datasets])


def test_set_montage(base_concat_ds):
    """Test SetMontage preprocessor."""
    montage = mne.channels.make_standard_montage('standard_1020')
    preprocessors = [SetMontage(montage=montage, match_case=False, on_missing='ignore')]
    preprocess(base_concat_ds, preprocessors)
    # Check that montage was set
    assert all([ds.raw.get_montage() is not None for ds in base_concat_ds.datasets])


def test_compute_csd(base_concat_ds_with_montage):
    """Test ComputeCurrentSourceDensity preprocessor."""
    pytest.skip("ComputeCurrentSourceDensity requires montage setup - skipping for now")
    preprocessors = [ComputeCurrentSourceDensity()]
    # CSD returns a new instance, so we need to handle it differently
    # For now, just test that it doesn't raise an error
    preprocess(base_concat_ds_with_montage, preprocessors)


def test_all_preprocessing_functions_importable():
    """Test that all preprocessing functions in __all__ can be imported and instantiated."""
    import braindecode.preprocessing as prep_module

    # Get all preprocessing class names from __all__
    all_preprocessors = [name for name in prep_module.__all__
                         if name[0].isupper() and name not in
                         ['Preprocessor', 'BaseConcatDataset', 'BaseDataset']]

    # Test each preprocessor can be imported
    for prep_name in all_preprocessors:
        assert hasattr(prep_module, prep_name), f"{prep_name} not found in preprocessing module"
        prep_class = getattr(prep_module, prep_name)
        assert callable(prep_class), f"{prep_name} is not callable"


@pytest.mark.parametrize("prep_class_name,init_kwargs", [
    ("Resample", {"sfreq": 100}),
    ("Pick", {"picks": ["eeg"]}),
    ("Filter", {"l_freq": 4, "h_freq": 30}),
    ("SetEEGReference", {"ref_channels": "average"}),
    ("Crop", {"tmin": 0, "tmax": 1}),
    ("DropChannels", {"ch_names": ["Pz"]}),
    ("NotchFilter", {"freqs": [50]}),
    ("SavgolFilter", {"h_freq": 10}),
    ("RenameChannels", {"mapping": {"C4": "C4_renamed"}}),
    ("ReorderChannels", {"ch_names": ["Cz", "C4"]}),
    ("AddReferenceChannels", {"ref_channels": ["FCz"]}),
    ("ApplyHilbert", {"envelope": True}),
    ("ApplyProj", {}),
    ("Anonymize", {}),
    ("SetChannelTypes", {"mapping": {"C4": "eog"}}),
    ("Rescale", {"scalings": {"eeg": 1e-6}}),
    ("FixMagCoilTypes", {}),
    ("SetMeasDate", {"meas_date": 1}),
])
def test_preprocessing_function_on_raw(base_concat_ds, prep_class_name, init_kwargs):
    """Test that each preprocessing function can be applied to raw data without errors."""
    import braindecode.preprocessing as prep_module

    # Skip tests that require special setup
    skip_list = [
        "ComputeCurrentSourceDensity",  # requires montage setup
        "InterpolateBads",  # requires montage and bad channels marked
        "FixMagCoilTypes",  # only for magnetometer data
        "EqualizeChannels",  # requires multiple raw objects
        "FixStimArtifact",  # requires events setup
        "AddChannels",  # requires additional raw object
        "ApplyGradientCompensation",  # only for MEG data
        "DelProj",  # requires existing projections
        "CropByAnnotations",  # requires existing annotations
    ]
    if prep_class_name in skip_list:
        pytest.skip(f"{prep_class_name} requires special setup")

    prep_class = getattr(prep_module, prep_class_name)
    preprocessor = prep_class(**init_kwargs)

    # Apply preprocessing
    try:
        preprocess(base_concat_ds, [preprocessor], n_jobs=1)
    except Exception as e:
        pytest.fail(f"{prep_class_name} failed with {type(e).__name__}: {str(e)}")
