import torch


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

    def transform(self, X):
        if not (len(X.shape) == 4):
            # (len(X.shape) == 4) characterizes the
            # spectrogramm of an epoch with several
            # channels.
            X = torch.stft(X, n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length,
                           window=torch.hann_window(self.n_fft))

        return self.policy(X, self.params)
        # if type = "axis_warp":
        #    return(warp_along_axis())


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

    def transform(self, X):
        if (len(X.shape) == 4):
            X = torch.istft(X,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.n_fft,
                            window=torch.hann_window(self.n_fft),
                            length=3000)
        else:
            X = torch.tensor(X)
        return self.policy(X, self.params)
