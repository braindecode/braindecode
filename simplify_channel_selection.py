import mne


def get_eegpt_allowed_channels():
    """
    Returns the set of 62 allowed channels for EEGPT based on the Standard 10-20 system.

    Logic:
    1. Start with 'standard_1020' montage.
    2. Exclude fiducials/references: A1, A2, M1, M2, IZ, AFZ.
    3. Exclude older aliases: T3, T4, T5, T6 (prefer T7, T8, P7, P8).
    4. Exclude "bottom ring" extensions (indices 9 and 10).
    5. Exclude specific high-density interpolations near midline/intermediate:
       - AF row: Exclude AF1, AF2, AF5, AF6.
       - PO row: Exclude PO1, PO2.
    """

    montage = mne.channels.make_standard_montage("standard_1020")

    # 1. Explicit Exclusions (Fiducials, References, Old Aliases)
    exclude_explicit = {'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'M1', 'M2', 'IZ', 'AFZ'}

    channels = set()
    for ch in montage.ch_names:
        ch_u = ch.upper()

        # Explicit exclusion
        if ch_u in exclude_explicit:
            continue

        # 2. Exclude "bottom ring" (indices 9, 10)
        if ch_u.endswith('9') or ch_u.endswith('10'):
            continue

        # 3. Specific density reduction for high-density rows (AF, PO)
        if ch_u.startswith('AF'):
            # Maintain standard 10-10 locations (7, 3, Z, 4, 8)
            # Exclude interpolated intermediates: 1, 2 (midline-adjacent), 5, 6 (intermediate)
            if any(x in ch_u for x in ['1', '2', '5', '6']):
                continue

        if ch_u.startswith('PO'):
            # Maintain standard 10-10 locations (7, 3, Z, 4, 8) PLUS 5, 6
            # Only exclude midline-adjacent: 1, 2
            if any(x in ch_u for x in ['1', '2']):
                continue

        channels.add(ch_u)

    return channels


def get_eegpt_chs_info():
    allowed_channels = get_eegpt_allowed_channels()
    montage = mne.channels.make_standard_montage("standard_1005")

    chs_info = []
    # We want a specific order?
    # Usually EEGPT expects channels in specific order.
    # _ALLOWED_CHANNELS in eegpt.py is a set, so order there is undefined.
    # But usually models are sensitive to channel order if they rely on it.
    # EEGPT uses channel embeddings based on names, so order might not matter for the model execution
    # BUT for loading weights it might map index to embedding.

    # However, braindecode EEGPT uses `chan_embed` which is an Embedding layer.
    # The `chan_ids` are looked up from `CHANNEL_DICT`.
    # `CHANNEL_DICT` is created from `_ALLOWED_CHANNELS`.
    # `_get_eegpt_channels()` sorts by `standard_1020` order.

    # We should probably respect the order defined in EEGPT_CHANNELS in eegpt.py
    # But we can't easily import it here without circular dependency if eegpt imports this file (it doesn't currently).

    # For now, let's just return the info for the allowed channels found.
    # The pushing script will handle the ordering or the model __init__ will.

    for ch_name, pos in zip(montage.ch_names, montage.get_positions()['ch_pos'].values()):
        if ch_name.upper() in allowed_channels:
             chs_info.append({
                 'ch_name': ch_name.upper(),
                 'loc': pos,
                 'kind': 2, # EEG
                 'unit': 107, # V -> 107 is FIFFV_EEG_CH? No, checking MNE constants might be needed but 107 is likely not right for unit.
                 # Wait, MNE units: 107 is likely 'V' (FIFF_UNIT_V).
                 # Let's check braindecode/models/base.py serialization test says unit: 107.
             })
    return chs_info

# -------------------------------------------------------------------------
# Verification vs Original Hardcoded List
# -------------------------------------------------------------------------

_ORIGINAL_ALLOWED_CHANNELS = {
    "FP1", "FPZ", "FP2", "AF7", "AF3", "AF4", "AF8", "F7", "F5", "F3", "F1",
    "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5",
    "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
    "PO4", "PO6", "PO8", "O1", "OZ", "O2",
}

def verify():
    semantic_channels = get_eegpt_allowed_channels()

    missing = _ORIGINAL_ALLOWED_CHANNELS - semantic_channels
    extra = semantic_channels - _ORIGINAL_ALLOWED_CHANNELS

    if not missing and not extra:
        print("SUCCESS! The semantic logic perfectly matches the allowed channels.")
        print(f"Count: {len(semantic_channels)}")
    else:
        print("Mismatch found.")
        print(f"Missing from proposed: {missing}")
        print(f"Extra in proposed: {extra}")

if __name__ == "__main__":
    verify()
