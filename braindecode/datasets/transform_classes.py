import torch
import numpy as np


class TransformFFT:

    def __init__(self, policy, params={},
                 fft_args={"n_fft": 512, "hop_length": 256,
                           "win_length": 512}):
        self.policy = policy
        self.params = params
        self.n_fft = fft_args["n_fft"]
        self.hop_length = fft_args["hop_length"]
        self.win_length = fft_args["win_length"]

    def fit(self, X):
        pass

    def transform(self, datum):
        X = datum.X
        if not (len(X.shape) == 4):
            # (len(X.shape) == 4) characterizes the
            # spectrogramm of an epoch with several
            # channels.
            datum.X = torch.stft(X, n_fft=self.n_fft,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 window=torch.hann_window(self.n_fft))

        datum = self.policy(datum, self.params)
        return datum


class TransformSignal:

    def __init__(self, policy, params={},
                 fft_args={"n_fft": 512, "hop_length": 256,
                           "win_length": 512}):
        self.params = params
        self.policy = policy
        self.n_fft = fft_args["n_fft"]
        self.hop_length = fft_args["hop_length"]
        self.win_length = fft_args["win_length"]

    def fit(self, X):
        pass

    def __call__(self, datum):
        X = datum.X
        if (len(X.shape) == 4):
            datum.X = torch.istft(X,
                                  n_fft=self.n_fft,
                                  hop_length=self.hop_length,
                                  win_length=self.n_fft,
                                  window=torch.hann_window(self.n_fft),
                                  length=3000)
        else:
            if type(X).__module__ == np.__name__:
                datum.X = torch.tensor(X)
        datum = self.policy(datum, self.params)
        return datum
        # TODO : update with __call__ Utilise l'objet compose de torchvision.
