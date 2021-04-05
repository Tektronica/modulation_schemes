import matplotlib.pyplot as plot
import numpy as np
import sys
from scipy import signal

file = '1111'
m = 0.2
TYPE = 'fsk'
freq = 10
freqs = 2

if len(sys.argv) > 1:
    file = str(sys.argv[1])

if len(sys.argv) > 2:
    TYPE = (sys.argv[2])

if len(sys.argv) > 3:
    freq = int(sys.argv[3])

Fs = 150.0  # sampling rate
Ts = 1.0 / Fs  # sampling interval

t = np.arange(0, 2, Ts)

if TYPE == 'fsk':
    bit_arr = np.array([5, 5, -5, 5, -5])
    samples_per_bit = 2 * Fs / bit_arr.size
    dd = np.repeat(bit_arr, samples_per_bit)
    y = np.sin(2 * np.pi * (freq + dd) * t)

elif TYPE == 'psk':
    bit_arr = np.array([180, 180, 0, 180, 0])
    samples_per_bit = 2 * Fs / bit_arr.size
    dd = np.repeat(bit_arr, samples_per_bit)
    y = np.sin(2 * np.pi * freq * t + (np.pi * dd / 180))

else:
    bit_arr = np.array([1, 0, 1, 1, 0])
    samples_per_bit = 2 * Fs / bit_arr.size
    dd = np.repeat(bit_arr, samples_per_bit)
    y = dd * np.sin(2 * np.pi * freq * t)

n = len(y)  # length of the signal
k = np.arange(n)
T = n / Fs
frq = k / T  # two sides frequency range
frq = frq[range(n // 2)]  # one side frequency range
Y = np.fft.fft(y) / n  # fft computing and normalization
Y = Y[range(n // 2)]

fig, myplot = plot.subplots(2, 1)
myplot[0].plot(t, y)
myplot[0].set_xlabel('Time')
myplot[0].set_ylabel('Amplitude')

myplot[1].plot(frq, abs(Y), 'r')  # plotting the spectrum
myplot[1].set_xlabel('Freq (Hz)')
myplot[1].set_ylabel('|Y(freq)|')

# plot.savefig(file)
plot.show()
print("Saving to ", file)
