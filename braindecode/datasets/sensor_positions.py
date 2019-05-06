import numpy as np
import math


CHANNEL_10_20_APPROX = (
    "angle",
    ("Fpz", (0.000, 4.000)),
    ("Fp1", (-3.500, 3.500)),
    ("Fp2", (3.500, 3.500)),
    ("AFp3h", (-1.000, 3.500)),
    ("AFp4h", (1.000, 3.500)),
    ("AF7", (-4.000, 3.000)),
    ("AF3", (-2.000, 3.000)),
    ("AFz", (0.000, 3.000)),
    ("AF4", (2.000, 3.000)),
    ("AF8", (4.000, 3.000)),
    ("AFF5h", (-2.500, 2.500)),
    ("AFF1", (-0.500, 2.500)),
    ("AFF2", (0.500, 2.500)),
    ("AFF6h", (2.500, 2.500)),
    ("F7", (-4.000, 2.000)),
    ("F5", (-3.000, 2.000)),
    ("F3", (-2.000, 2.000)),
    ("F1", (-1.000, 2.000)),
    ("Fz", (0.000, 2.000)),
    ("F2", (1.000, 2.000)),
    ("F4", (2.000, 2.000)),
    ("F6", (3.000, 2.000)),
    ("F8", (4.000, 2.000)),
    ("FFT7h", (-3.500, 1.500)),
    ("FFC5h", (-2.500, 1.500)),
    ("FFC3h", (-1.500, 1.500)),
    ("FFC1h", (-0.500, 1.500)),
    ("FFC2h", (0.500, 1.500)),
    ("FFC4h", (1.500, 1.500)),
    ("FFC6h", (2.500, 1.500)),
    ("FFT8h", (3.500, 1.500)),
    ("FT9", (-5.000, 1.000)),
    ("FT7", (-4.000, 1.000)),
    ("FC5", (-3.000, 1.000)),
    ("FC3", (-2.000, 1.000)),
    ("FC1", (-1.000, 1.000)),
    ("FCz", (0.000, 1.000)),
    ("FC2", (1.000, 1.000)),
    ("FC4", (2.000, 1.000)),
    ("FC6", (3.000, 1.000)),
    ("FT8", (4.000, 1.000)),
    ("FT10", (5.000, 1.000)),
    ("FTT9h", (-4.500, 0.500)),
    ("FTT7h", (-3.500, 0.500)),
    ("FCC5h", (-2.500, 0.500)),
    ("FCC3h", (-1.500, 0.500)),
    ("FCC1h", (-0.500, 0.500)),
    ("FCC2h", (0.500, 0.500)),
    ("FCC4h", (1.500, 0.500)),
    ("FCC6h", (2.500, 0.500)),
    ("FTT8h", (3.500, 0.500)),
    ("FTT10h", (4.500, 0.500)),
    ("M1", (-5.000, 0.000)),
    # notsure if correct:
    ("T9", (-4.500, 0.000)),
    ("T7", (-4.000, 0.000)),
    ("C5", (-3.000, 0.000)),
    ("C3", (-2.000, 0.000)),
    ("C1", (-1.000, 0.000)),
    ("Cz", (0.000, 0.000)),
    ("C2", (1.000, 0.000)),
    ("C4", (2.000, 0.000)),
    ("C6", (3.000, 0.000)),
    ("T8", (4.000, 0.000)),
    ("T10", (4.500, 0.000)),
    ("M2", (5.000, 0.000)),
    ("TTP7h", (-3.500, -0.500)),
    ("CCP5h", (-2.500, -0.500)),
    ("CCP3h", (-1.500, -0.500)),
    ("CCP1h", (-0.500, -0.500)),
    ("CCP2h", (0.500, -0.500)),
    ("CCP4h", (1.500, -0.500)),
    ("CCP6h", (2.500, -0.500)),
    ("TTP8h", (3.500, -0.500)),
    ("TP7", (-4.000, -1.000)),
    ("CP5", (-3.000, -1.000)),
    ("CP3", (-2.000, -1.000)),
    ("CP1", (-1.000, -1.000)),
    ("CPz", (0.000, -1.000)),
    ("CP2", (1.000, -1.000)),
    ("CP4", (2.000, -1.000)),
    ("CP6", (3.000, -1.000)),
    ("TP8", (4.000, -1.000)),
    ("TPP9h", (-4.500, -1.500)),
    ("TPP7h", (-3.500, -1.500)),
    ("CPP5h", (-2.500, -1.500)),
    ("CPP3h", (-1.500, -1.500)),
    ("CPP1h", (-0.500, -1.500)),
    ("CPP2h", (0.500, -1.500)),
    ("CPP4h", (1.500, -1.500)),
    ("CPP6h", (2.500, -1.500)),
    ("TPP8h", (3.500, -1.500)),
    ("TPP10h", (4.500, -1.500)),
    ("P9", (-5.000, -2.000)),
    ("P7", (-4.000, -2.000)),
    ("P5", (-3.000, -2.000)),
    ("P3", (-2.000, -2.000)),
    ("P1", (-1.000, -2.000)),
    ("Pz", (0.000, -2.000)),
    ("P2", (1.000, -2.000)),
    ("P4", (2.000, -2.000)),
    ("P6", (3.000, -2.000)),
    ("P8", (4.000, -2.000)),
    ("P10", (5.000, -2.000)),
    ("PPO9h", (-4.500, -2.500)),
    ("PPO5h", (-3.000, -2.500)),
    ("PPO1", (-0.650, -2.500)),
    ("PPO2", (0.650, -2.500)),
    ("PPO6h", (3.000, -2.500)),
    ("PPO10h", (4.500, -2.500)),
    ("PO9", (-5.000, -3.000)),
    ("PO7", (-4.000, -3.000)),
    ("PO5", (-3.000, -3.000)),
    ("PO3", (-2.000, -3.000)),
    ("PO1", (-1.000, -3.000)),
    ("POz", (0.000, -3.000)),
    ("PO2", (1.000, -3.000)),
    ("PO4", (2.000, -3.000)),
    ("PO6", (3.000, -3.000)),
    ("PO8", (4.000, -3.000)),
    ("PO10", (5.000, -3.000)),
    ("POO9h", (-4.500, -3.250)),
    ("POO3h", (-2.000, -3.250)),
    ("POO4h", (2.000, -3.250)),
    ("POO10h", (4.500, -3.250)),
    ("O1", (-2.500, -3.750)),
    ("Oz", (0.000, -3.750)),
    ("O2", (2.500, -3.750)),
    ("OI1h", (1.500, -4.250)),
    ("OI2h", (-1.500, -4.250)),
    ("I1", (1.000, -4.500)),
    ("Iz", (0.000, -4.500)),
    ("I2", (-1.000, -4.500)),
)


def get_channelpos(channame, chan_pos_list):
    if chan_pos_list[0] == "angle":
        return get_channelpos_from_angle(channame, chan_pos_list[1:])
    elif chan_pos_list[0] == "cartesian":
        channame = channame.lower()
        for name, coords in chan_pos_list[1:]:
            if name.lower() == channame:
                return coords[0], coords[1]
        return None
    else:
        raise ValueError(
            "Unknown first element "
            "{:s} (should be type of positions)".format(chan_pos_list[0])
        )


def get_channelpos_from_angle(channame, chan_pos_list=CHANNEL_10_20_APPROX):
    """Return the x/y position of a channel.

    This method calculates the stereographic projection of a channel
    from ``CHANNEL_10_20``, suitable for a scalp plot.

    Parameters
    ----------
    channame : str
        Name of the channel, the search is case insensitive.

    chan_pos_list=CHANNEL_10_20_APPROX,
    interpolation='bilinear'

    Returns
    -------
    x, y : float or None
        The projected point on the plane if the point is known,
        otherwise ``None``

    Examples
    --------

    >>> plot.get_channelpos_from_angle('C2')
    (0.1720792096741632, 0.0)
    >>> # the channels are case insensitive
    >>> plot.get_channelpos_from_angle('c2')
    (0.1720792096741632, 0.0)
    >>> # lookup for an invalid channel
    >>> plot.get_channelpos_from_angle('foo')
    None

    """
    channame = channame.lower()
    for i in chan_pos_list:
        if i[0].lower() == channame:
            # convert the 90/4th angular position into x, y, z
            p = i[1]
            x, y = _convert_2d_angle_to_2d_coord(*p)
            return x, y
    return None


def _convert_2d_angle_to_2d_coord(a, b):
    # convert the 90/4th angular position into x, y, z
    ea, eb = a * (90 / 4), b * (90 / 4)
    ea = ea * math.pi / 180
    eb = eb * math.pi / 180
    x = math.sin(ea) * math.cos(eb)
    y = math.sin(eb)
    z = math.cos(ea) * math.cos(eb)
    # Calculate the stereographic projection.
    # Given a unit sphere with radius ``r = 1`` and center at
    # the origin. Project the point ``p = (x, y, z)`` from the
    # sphere's South pole (0, 0, -1) on a plane on the sphere's
    # North pole (0, 0, 1).
    #
    # The formula is:
    #
    # P' = P * (2r / (r + z))
    #
    # We changed the values to move the point of projection
    # further below the south pole
    mu = 1 / (1.3 + z)
    x *= mu
    y *= mu
    return x, y
