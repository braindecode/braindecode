from scipy.optimize import leastsq
import numpy as np


def err_fn_sin(p, x, y):
    return (y - fit_fn_sin(x, *p)).flat


def err_fn_lin(p, x, y):
    return (y - fit_fn_lin(x, *p)).flat


def fit_fn_lin(x, *kwargs):
    return kwargs[0] + kwargs[1] * x


def fit_fn_sin(x, *kwargs):
    freqs = kwargs[0]
    amps = kwargs[1]
    phases = kwargs[2]
    offset = kwargs[3]
    sig = np.zeros((len(x))) + offset
    sig += amps * np.cos(x * freqs + phases)
    return sig


def signal_fit(signals, fs):
    """Fits sinusoid and linear function to signals
    see sinfit.fit_fn_sin and sinfit.fit_fn_lin

    Parameters
    ----------
    signals : numpy array
        [FxCxTx1] Filters x Channels x Time x 1
    fs : float
        Sampling frequency

    Returns
    -------
    params_sin : numpy array
        [FxCx4] Parameters of sinusoid fit
        Parameters are: Frequency,Amplitude,Phase,DCOffset
    params_lin : numpy array
        [FxCx2] Parameters of sinusoid fit
        Parameters are: Frequency,Amplitude,Phase,DCOffset
    err_sin : numpy array
        [FxCx1] MSE for sinusoid fit
    err_lin : numpy array
        [FxCx1] MSE for linear fit
    """
    params_sin = []
    params_lin = []

    err_sin = []
    err_lin = []

    freqs = np.fft.rfftfreq(signals.shape[2], d=1.0 / fs)[1:]
    x = np.linspace(0, signals.shape[2] / fs, signals.shape[2]) * 2 * np.pi
    for filt in range(signals.shape[0]):
        params_sin_tmp = []
        params_lin_tmp = []

        err_sin_tmp = []
        err_lin_tmp = []
        for ch in range(signals.shape[1]):
            X_tmp = signals[filt, ch].squeeze()
            fft_X = np.fft.rfft(X_tmp, axis=0)
            amps_mean = np.abs(fft_X)[1:]
            phases_mean = np.angle(fft_X)[1:]
            offset = X_tmp.mean()

            sort = np.argsort(amps_mean)[::-1][0]
            p0 = [freqs[sort], amps_mean[sort], phases_mean[sort], offset]

            fit_sin_ch = leastsq(err_fn_sin, p0, args=(x, X_tmp), maxfev=100000)
            fit_lin_ch = leastsq(err_fn_lin, [0, 0], args=(x, X_tmp), maxfev=100000)

            err_sin_ch = np.square(fit_fn_sin(x, *fit_sin_ch[0]) - X_tmp).mean()
            err_lin_ch = np.square(fit_fn_lin(x, *fit_lin_ch[0]) - X_tmp).mean()

            params_sin_tmp.append(fit_sin_ch[0])
            params_lin_tmp.append(fit_lin_ch[0])
            err_sin_tmp.append(err_sin_ch)
            err_lin_tmp.append(err_lin_ch)
        params_sin.append(params_sin_tmp)
        params_lin.append(params_lin_tmp)
        err_sin.append(err_sin_tmp)
        err_lin.append(err_lin_tmp)
    params_sin = np.asarray(params_sin)
    params_lin = np.asarray(params_lin)
    err_sin = np.asarray(err_sin)
    err_lin = np.asarray(err_lin)

    return params_sin, params_lin, err_sin, err_lin
