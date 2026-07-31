import numpy as np

def sin(x, freq, amp, phase):
    return amp * np.sin(2.0 * np.pi * (freq * x + phase))