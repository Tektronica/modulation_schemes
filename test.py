import numpy as np

import matplotlib.pyplot as plt


N = 48000
Fs = 80000

xt = np.arange(0, N, 1) / Fs
ct = 1 * np.cos(2 * np.pi * 1000 * xt)

# remove DC offset
ct -= np.mean(ct)

# window
w = np.blackman(N)

amplitude_correction_factor = 1/np.mean(w)
print(amplitude_correction_factor)

if (N % 2) == 0:
    # for even values of N: length is (N / 2) + 1
    length = int(N / 2) + 1
else:
    # for odd values of N: length is (N + 1) / 2
    length = int((N + 2) / 2)

# https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
# function does not compute the negative frequency terms
# divide by N samples to normalize
yf_fft = np.fft.fft(ct * w) / length

yf_rfft = yf_fft[:length]*amplitude_correction_factor
xf_fft = np.linspace(0.0, Fs, N)
xf_rfft = np.linspace(0.0, Fs / 2, length)


figure = plt.figure(figsize=(1, 1))  # look into Figure((5, 4), 75)
ax1 = figure.add_subplot(211)
ax2 = figure.add_subplot(212)

temporal, = ax1.plot(xt, ct, linestyle='-')
spectral, = ax2.plot(xf_rfft, np.abs(yf_rfft), color='#C02942')

plt.show()
