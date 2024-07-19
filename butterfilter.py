import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, butter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Sample rate and desired cutoff frequencies (in Hz).
fs = 16000.0
lowcut = 400.0
highcut = 5500.0

# Plot the frequency response for a few different orders.
plt.figure(1)
plt.clf()

b, a = butter_bandpass(lowcut, highcut, fs, order=9)
w, h = freqz(b, a, worN=2000)
plt.plot((fs * 0.5 / np.pi) * w, abs(h))

plt.title('Band-Pass-Filter Butterworth')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')

plt.axvline(400, color='red') # cutoff frequency
plt.axvline(5500, color='red') # cutoff frequency
