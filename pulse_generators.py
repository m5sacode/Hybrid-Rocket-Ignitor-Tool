import numpy as np
import matplotlib.pyplot as plt

def smooth_pulse(
    x,
    duration=1.0,      # total duration (flat + rise + fall)
    rise_time=0.1,     # rise duration
    fall_time=0.1,     # fall duration
    area=1.0           # desired integral under the curve
):
    """
    Generates a smooth pulse that rises smoothly, stays flat, and falls smoothly,
    with a controllable total integral (area), duration, rise time, and fall time.

    Parameters
    ----------
    x : np.ndarray
        Input time vector.
    duration : float
        Total pulse duration (seconds or arbitrary units).
    rise_time : float
        Time for the rising edge (smoothly from 0→1).
    fall_time : float
        Time for the falling edge (smoothly from 1→0).
    area : float
        Desired integral (area under the curve).

    Returns
    -------
    y : np.ndarray
        Pulse waveform.
    """

    # Ensure parameters are consistent
    flat_time = max(duration - rise_time - fall_time, 0)
    start = 0
    rise_end = start + rise_time
    flat_end = rise_end + flat_time
    fall_end = flat_end + fall_time

    # Smoothstep (cubic Hermite)
    def smoothstep(edge0, edge1, x):
        t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
        return t * t * (3 - 2 * t)

    # Construct waveform: rise → flat → fall
    rise = smoothstep(start, rise_end, x)
    fall = 1 - smoothstep(flat_end, fall_end, x)
    y = np.minimum(rise, fall)

    # Normalize so that the integral (area) = desired 'area'
    current_area = np.trapz(y, x)
    if current_area > 0:
        y *= area / current_area

    return y


# Example usage
x = np.linspace(0, 2, 2000)
y = smooth_pulse(x, duration=2.0, rise_time=0.7, fall_time=0.7, area=2.0)

plt.plot(x, y)
plt.title("Smooth Pulse with Controlled Area, Duration, Rise/Fall")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

def smooth_rise_linear_fall(
    x,
    duration=1.0,     # total pulse duration
    rise_time=0.2,    # time for rising edge
    area=1.0          # desired total integral (area under the curve)
):
    """
    Generates a pulse that rises smoothly (using smoothstep) and then linearly falls to zero,
    with controllable total duration, rise time, and area.

    Parameters
    ----------
    x : np.ndarray
        Input time vector.
    duration : float
        Total pulse duration (same units as x).
    rise_time : float
        Time for the rising edge (smooth transition from 0→1).
    area : float
        Desired total integral (area under curve).

    Returns
    -------
    y : np.ndarray
        Pulse waveform.
    """

    # Check validity
    if rise_time >= duration:
        raise ValueError("rise_time must be smaller than total duration")

    # Define key time points
    start = 0
    rise_end = start + rise_time

    # Smooth rise (cubic Hermite)
    def smoothstep(edge0, edge1, x):
        t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
        return t * t * (3 - 2 * t)

    # Compute rise segment
    rise = smoothstep(start, rise_end, x)

    # Compute linear fall starting at rise_end → 0 at duration
    slope = -1.0 / (duration - rise_end)
    fall = 1.0 + slope * (x - rise_end)
    fall = np.clip(fall, 0, None)

    # Combine both parts
    y = np.where(x < rise_end, rise, fall)

    # Normalize total area
    current_area = np.trapz(y, x)
    if current_area > 0:
        y *= area / current_area

    return y


# Example usage
# x = np.linspace(0, 2, 2000)
# y = smooth_rise_linear_fall(x, duration=2.0, rise_time=0.4, area=1.5)
#
# plt.plot(x, y)
# plt.title("Smooth Rise + Linear Fall Pulse (Controlled Area)")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.show()
