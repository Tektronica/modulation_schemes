import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

N = 48000  # number of samples
Fs = 80000  # sample rate
fm = 500  # message signal frequency
phase = 0  # phase of message signal

N_range = np.arange(0, N, 1)  # sample base
xt = N_range / Fs  # time base
p = Fs / fm  # samples per period
Am = 1  # amplitude

waveform = 'triangle'

if waveform == 'sine':
    mt = 1 * np.cos(2 * np.pi * 1000 * xt)
    mt_ = np.sin(np.pi * fm * xt) * np.cos(np.pi * fm * xt + fm)

elif waveform == 'triangle':
    # Triangle
    # https://en.wikipedia.org/wiki/Triangle_wave#Modulo_operation
    T = Fs / fm

    # message signal (modulo operation)
    N_phase_shifted = N_range + int(T * phase / 360)

    mt = ((4 * Am) / T) * np.abs(((N_phase_shifted - T / 4) % T) - T / 2) - 1

    # integral of triangle ---------------------------------------------------------------------------------------------
    integral_shift = N_phase_shifted + T / 4
    print(fm, T)

    def first(x, T):
        t = (x % (T))
        return fm / 10000 * (2 / T * t ** 2 - t)

    def second(x, T):
        t = (x % (T))
        return fm / 10000 * (-2 / T * t ** 2 + 3 * t - T)

    mt_ = np.where((integral_shift % T) < T / 2, first(integral_shift, T), second(integral_shift, T))

    st = 1 * np.cos(2 * np.pi * 1000 * xt + 5 * mt_)

    # square wave ------------------------------------------------------------------------------------------------------
    mt__ = np.where(((integral_shift - T/2) % T) < T / 2, 1, -1)

    # scipy integration of trangle wave --------------------------------------------------------------------------------
    mt___ = integrate.cumtrapz(mt) / 100 - 1

    print(p)
    n = 200
    print(N_range[n], xt[n], mt___[n])
    n = 400
    print(N_range[n], xt[n], mt___[n])
    n = 800
    print(N_range[n], xt[n], mt___[n])

elif waveform == 'square':
    mt = np.where((((N_range % p) + p) % p) < p / 2, 1, 0)
    N_range_phase_shift = N_range + p * 90 / 360
    print(N_range)
    mt_ = 4 / p * np.abs((((N_range_phase_shift - p / 4) % p) + p) % p - p / 2) - 1
else:
    raise ValueError("Invalid waveform type selected!")

# remove DC offset
mt_ -= np.mean(mt_)

# window
w = np.blackman(N)

amplitude_correction_factor = 1 / np.mean(w)
print(amplitude_correction_factor)

if (N % 2) == 0:
    # for even values of N: length is (N / 2) + 1
    fft_length = int(N / 2) + 1
else:
    # for odd values of N: length is (N + 1) / 2
    fft_length = int((N + 2) / 2)

# perform FFT ----------------------------------------------------------------------------------------------------------
# https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
# function does not compute the negative frequency terms
# divide by N samples to normalize
yf_fft = np.fft.fft(mt_ * w) / fft_length

yf_rfft = yf_fft[:fft_length] * amplitude_correction_factor
xf_fft = np.linspace(0.0, Fs, N)
xf_rfft = np.linspace(0.0, Fs / 2, fft_length)

# spectral y scale parameters ------------------------------------------------------------------------------------------
n_tick = 5
yf_data_peak = round(abs(max(yf_rfft)), 1)
yf_tick = round(np.ceil((yf_data_peak / (n_tick - 2)) * 10) / 10, 1)

# https://stackoverflow.com/a/66805331/3382269
yf_btm = -yf_tick
yf_top = round((n_tick - 2) * yf_tick, 1)

print("FFT y-scale parameters:")
print('yf_data_peak:', yf_data_peak, 'yf_btm:', yf_btm, 'yf_top:', yf_top, 'yf_tick:', yf_tick)

# plot -----------------------------------------------------------------------------------------------------------------
figure = plt.figure()  # look into Figure((5, 4), 75)
ax1 = figure.add_subplot(211)
ax2 = figure.add_subplot(212)

temporal, = ax1.plot(xt, mt, '-')  # Triangle
temporal_2, = ax1.plot(xt, mt_, '--')  # integral of triangle
temporal_3, = ax1.plot(xt, mt__, '--')  # square
# temporal_4, = ax1.plot(xt[1:], mt___, '--')
# temporal_5, = ax1.plot(xt, st, '--')

ax1.set_xlim((0, 4 / fm))

spectral, = ax2.plot(xf_rfft, np.abs(yf_rfft), color='#C02942')

plt.show()
