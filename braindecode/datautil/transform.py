from typing import Dict
import torch
import torchaudio

from .transform_side_funcs import datautil

class Policy:

    def __init__(type: string, args: Dict):
        self.type = type
        self.args = args

    def apply(self, spectrogram):
        if type == "axis_mask":
            return(mask_along_axis(spectrogram, self.args["mask_start"], self.args["mask_end"], self.args["mask_value"], self.args["axis"]))
        # if type = "axis_warp":
        #    return(warp_along_axis())


class WindowTransformer:
    def __init__(policy_list: Policy):
        self.policy_list = policy_list
    
    def transform(self, window):
        n_fft = 512
        hop_length = 256
        win_length = n_fft
        window.get_data()
        spectrogram_list = torch.stft(window, n_fft=n_fft, 
                                      hop_length=hop_length,
                                      win_length=n_fft,
                                      window=torch.hann_window(n_fft),
                                      center=False)

        for policy in self.policy_list:
            spectrogram = policy.apply(spectrogram)

        augmented_window = torchaudio.functional.istft(spectrogram,    
                                                       n_fft=n_fft,
                                                       hop_length=hop_length,
                                                       win_length=n_fft,
                                                       window=torch.hann_window(n_fft),
                                                       center=False)
