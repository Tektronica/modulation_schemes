import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

N = 48000
Fs = 80000
fm = 100

N_range = np.arange(0, N, 1)
xt = N_range / Fs
p = Fs/fm
Am = 1

waveform = 'triangle'

if waveform == 'sine':
    ct = 1 * np.cos(2 * np.pi * 1000 * xt)
    ct_ = np.sin(np.pi * fm * xt) * np.cos(np.pi * fm * xt + fm)
elif waveform == 'triangle':
    ct = 4 / p * np.abs((((N_range - p / 4) % p) + p) % p - p / 2) - 1
    ct_ = np.where((((N_range % p) + p) % p) < p / 2, 2*fm*(N_range/Fs)**2 - 1, -2*fm*(N_range/Fs)**2 + 3)
    # ct_ = integrate.cumtrapz(ct)/100 - 1

elif waveform == 'square':
    ct = np.where((((N_range % p) + p) % p) < p / 2, 1, 0)
    N_range_phase_shift = N_range + p * 90/360
    print(N_range)
    ct_ = 4 / p * np.abs((((N_range_phase_shift - p / 4) % p) + p) % p - p / 2) - 1
else:
    raise ValueError("Invalid waveform type selected!")

# remove DC offset
# ct -= np.mean(ct)

# window
w = np.blackman(N)

amplitude_correction_factor = 1 / np.mean(w)
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

yf_rfft = yf_fft[:length] * amplitude_correction_factor
xf_fft = np.linspace(0.0, Fs, N)
xf_rfft = np.linspace(0.0, Fs / 2, length)

figure = plt.figure()  # look into Figure((5, 4), 75)
ax1 = figure.add_subplot(211)
ax2 = figure.add_subplot(212)

temporal, = ax1.plot(xt, ct, '-')
temporal_2, = ax1.plot(xt, ct_, '--')
spectral, = ax2.plot(xf_rfft, np.abs(yf_rfft), color='#C02942')

plt.show()
