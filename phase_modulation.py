import time
import pandas as pd
import numpy as np
from pathlib import Path
import threading
import datetime
import os
import re
from decimal import Decimal
import csv


########################################################################################################################
def _getFilepath(directory, fname):
    Path(directory).mkdir(parents=True, exist_ok=True)
    date = datetime.date.today().strftime("%Y%m%d")
    filename = f'{fname}_{date}'
    index = 0

    while os.path.isfile(f'{directory}/{filename}_{str(index).zfill(3)}.csv'):
        index += 1
    filename = filename + "_" + str(index).zfill(3)
    return f'{directory}/{filename}.csv'


# ######################################################################################################################
def write_to_csv(path, fname, header, *args):
    table = list(zip(*args))
    pathname = _getFilepath(path, fname)
    with open(pathname, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        if header:
            writer.writerow(header)
        for row in table:
            writer.writerow(row)


########################################################################################################################
def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(np.absolute(a) ** 2))


def getWindowLength(f0=10e3, fs=2.5e6, windfunc='blackman', error=0.1):
    """
    Computes the window length of the measurement. An error is expressed since the main lobe width is directly
    proportional to the number of cycles captured. The minimum value of M correlates to the lowest detectable frequency
    by the windowing function. For instance, blackman requires a minimum of 6 period cycles of the frequency of interest
    in order to express content of that lobe in the DFT. Sampling frequency does not play a role in the width of the
    lobe, only the resolution of the lobe.

    :param fc: carrier frequency
    :param fs: sampling frequency
    :param windfunc: "Rectangular", "Bartlett", "Hanning", "Hamming", "Blackman"
    :param error: 100% error suggests the lowest detectable frequency is the fundamental
    :return: window length of integer value (number of time series samples collected)
    """
    # lowest detectable frequency by window
    ldf = f0 * error

    if windfunc == 'Rectangular':
        M = int(fs / ldf)
    elif windfunc in ('Bartlett', 'Hanning', 'Hamming'):
        M = int(4 * (fs / ldf))
    elif windfunc == 'blackman':
        M = int(6 * (fs / ldf))
    else:
        raise ValueError('Not a valid windowing function.')

    return M


def windowed_fft(yt, Fs, N, windfunc='blackman'):
    # remove DC offset
    # yt -= np.mean(yt)

    # Calculate windowing function and its length ----------------------------------------------------------------------
    if windfunc == 'bartlett':
        w = np.bartlett(N)
    elif windfunc == 'hanning':
        w = np.hanning(N)
    elif windfunc == 'hamming':
        w = np.hamming(N)
    elif windfunc == 'blackman':
        w = np.blackman(N)
    else:
        w = np.kaiser(N)

    # Calculate amplitude correction factor after windowing ------------------------------------------------------------
    # https://stackoverflow.com/q/47904399/3382269
    amplitude_correction_factor = 1 / np.mean(w)

    # Calculate the length of the FFT ----------------------------------------------------------------------------------
    if (N % 2) == 0:
        # for even values of N: length is (N / 2) + 1
        fft_length = int(N / 2) + 1
    else:
        # for odd values of N: length is (N + 1) / 2
        fft_length = int((N + 2) / 2)

    """
    Compute the FFT of the signal Divide by the length of the FFT to recover the original amplitude. Note dividing 
    alternatively by N samples of the time-series data splits the power between the positive and negative sides. 
    However, we are only looking at one side of the FFT.
    """
    yf_fft = np.fft.fft(yt * w) / fft_length

    yf_rfft = yf_fft[:fft_length] * amplitude_correction_factor
    xf_fft = np.linspace(0.0, Fs, N)
    xf_rfft = np.linspace(0.0, Fs / 2, fft_length)

    return xf_fft, yf_fft, xf_rfft, yf_rfft, fft_length


class Modulators:
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, parent):
        self.frame = parent
        self.DUMMY_DATA = False  # can be toggled by the gui
        self.user_values_good = False  # Flag indicates user input for amplitude value is good (True)

        self.params = {'mode': 0, 'source': 0, 'amplitude': '', 'units': '',
                       'rms': 0, 'frequency': 0.0, 'error': 0.0,
                       'cycles': 0.0, 'filter': ''}
        self.data = {'xt': [0], 'yt': [0],
                     'xf': [0], 'yf': [0]}
        self.results = {'Amplitude': [], 'Frequency': [], 'RMS': [],
                        'THDN': [], 'THD': [], 'RMS NOISE': [],
                        'N': [], 'Fs': [], 'Aperture': []}

    # ------------------------------------------------------------------------------------------------------------------
    def start(self, user_input):
        self.params = user_input

        selection = self.params['mode']

        carrier_amplitude = self.params['carrier_amplitude']
        carrier_frequency = self.params['carrier_frequency']
        modulation_index = self.params['modulation_index']
        message_frequency = self.params['message_frequency']
        message_phase = self.params['message_phase']

        print(f"Carrier: {carrier_amplitude} @ {carrier_frequency} Hz")
        message = f"Modulation: {modulation_index} @ {message_frequency} Hz with :{message_phase} shift"
        print(f"\n{message} {'-' * (100 - len(message))}")

        try:
            if selection == 1:
                self.run_selected_function(selection)
            elif self.user_values_good:
                self.run_selected_function(selection)
            else:
                self.frame.error_dialog('\nCheck your parameters!')
            self.frame.btn_start.SetLabel('RUN')

        except ValueError as e:
            self.user_values_good = False
            message = 'finished with errors.'
            print(f"{message} {'-' * (100 - len(message))}")

            self.frame.flag_complete = True
            self.frame.btn_start.SetLabel('RUN')
            self.frame.error_dialog(e)
        else:
            self.user_values_good = False
            message = 'done'
            print(f"{message} {'-' * (100 - len(message))}\n")
            self.frame.flag_complete = True

    # ------------------------------------------------------------------------------------------------------------------
    def run_selected_function(self, selection):
        try:
            # run single
            if selection == 0:
                self.run_single(self.simulation)

            # run sweep
            elif selection == 1:
                self.run_continuous(self.simulation)

            else:
                print('Nothing happened.')

        except ValueError:
            raise

    # ------------------------------------------------------------------------------------------------------------------
    def run_single(self, func):
        print('Running Single Measurement.')
        self.frame.toggle_controls()
        self.frame.flag_complete = False
        try:
            func()
        except ValueError:
            self.frame.toggle_controls()
            raise

        self.frame.toggle_controls()
        self.frame.flag_complete = True

    def run_continuous(self, func):
        print('Running a continuous run!')
        self.frame.flag_complete = False
        t = threading.currentThread()
        setup = True
        while getattr(t, "do_run", True):
            try:
                func()
                setup = False
                time.sleep(0.1)
            except ValueError:
                raise
        print('Ending continuous run_source process.')

    def simulation(self):
        WINDOW_FUNC = 'blackman'

        sample_rate = self.params['sample_rate']
        main_lobe_error = self.params['main_lobe_error']
        modulation_type = self.params['modulation_type']
        waveform_type = self.params['waveform_type']
        carrier_amplitude = self.params['carrier_amplitude']
        carrier_frequency = self.params['carrier_frequency']  # fc
        modulation_index = self.params['modulation_index']
        message_frequency = self.params['message_frequency']  # fm
        message_phase = self.params['message_phase']

        # time base ----------------------------------------------------------------------------------------------------
        N = getWindowLength(f0=message_frequency, fs=sample_rate, windfunc=WINDOW_FUNC, error=main_lobe_error)

        runtime = N / sample_rate
        N_range = np.arange(0, N, 1)
        xt = N_range / sample_rate

        # Compute message ----------------------------------------------------------------------------------------------
        if waveform_type == 'sine':
            # message signal
            mt = np.cos(2 * np.pi * message_frequency * xt + np.deg2rad(message_phase))
            # integral of message signal
            mt_ = np.sin(np.pi * message_frequency * xt) * np.cos(np.pi * message_frequency * xt + message_phase)

        elif waveform_type == 'triangle':
            # https://en.wikipedia.org/wiki/Triangle_wave#Modulo_operation
            p = sample_rate / message_frequency

            # message signal (modulo operation)
            applied_phase_shift = N_range + int(p * message_phase/360)
            mt = 4 / p * np.abs((((applied_phase_shift - p / 4) % p) + p) % p - p / 2) - 1

            # integral of message signal
            mt_ = np.sin(np.pi * message_frequency * xt) * np.cos(np.pi * message_frequency * xt + message_phase)

        elif waveform_type == 'square':
            p = sample_rate / message_frequency
            applied_phase_shift = N_range + int(p * message_phase/360)

            # message signal
            mt = np.where((((N_range % p) + p) % p) < p/2, 1, 0)

            # integral of message signal (square wave integral is a triangle wave (modulo operation) with 90 phase lag)
            corrected_phase_shift = applied_phase_shift + int(p * 90/360)
            mt_ = 4 / p * np.abs((((corrected_phase_shift - p / 4) % p) + p) % p - p / 2) - 1
        else:
            raise ValueError("Invalid waveform type selected!")

        # waveform data ------------------------------------------------------------------------------------------------
        # https://user.eng.umd.edu/~tretter/commlab/c6713slides/ch8.pdf

        if modulation_type == 0:
            # amplitude modulation -------------------------------------------------------------------------------------
            ct = carrier_amplitude * np.cos(2 * np.pi * carrier_frequency * xt)
            st = ct * mt

            bw = 2 * message_frequency

        elif modulation_type == 1:
            # frequency modulation -------------------------------------------------------------------------------------
            # In FM, the angle is directly proportional to the integral of m(t)
            kf = 1  # frequency deviation constant in rad/volt

            wm = 2.0 * np.pi * message_frequency
            pi_f = np.pi * message_frequency
            Am = (modulation_index * wm) / kf

            st = carrier_amplitude * np.cos(2 * np.pi * carrier_frequency * xt + kf * Am / pi_f * mt_)

            freq_deviation = kf * modulation_index  # frequency deviation
            beta = freq_deviation / message_frequency  # modulation index

            bw = 2 * message_frequency * (modulation_index + 1)

        elif modulation_type == 2:
            # phase modulation -------------------------------------------------------------------------------------
            # In PM, the angle is directly proportional to m(t)
            kp = 1  # frequency deviation constant in rad/volt

            st = carrier_amplitude * np.cos(2 * np.pi * carrier_frequency * xt + kp * modulation_index * mt)

            freq_deviation = kp * modulation_index
            beta = freq_deviation / message_frequency  # modulation index

            bw = 2 * message_frequency * (modulation_index + 1)

        else:
            raise ValueError("Invalid modulation type selected!")

        self.fft(xt, st, mt, runtime, bw, carrier_frequency, message_frequency, sample_rate, N, WINDOW_FUNC)

    def fft(self, xt, yt, mt, runtime, bw, fc, fm, Fs, N, WINDOW_FUNC='blackman'):
        yrms = rms_flat(yt)

        # FFT ==========================================================================================================
        xf_fft, yf_fft, xf_rfft, yf_rfft, fft_length = windowed_fft(yt, Fs, N, WINDOW_FUNC)
        data = {'x': xt, 'y': yt, 'bw': bw, 'mt': mt, 'xf': xf_rfft, 'ywf': yf_rfft, 'fft_length': fft_length,
                'RMS': yrms, 'N': N, 'runtime': runtime, 'Fs': Fs, 'fc': fc, 'fm': fm}

        # save measurement to csv --------------------------------------------------------------------------------------
        header = ['xt', 'yt', 'xf', 'yf']
        write_to_csv('results/history', 'modulation_schemes', header, xt, yt, xf_fft, yf_fft)

        # plot and report ----------------------------------------------------------------------------------------------
        report = {'yrms': yrms, 'bw': bw}
        self.frame.results_update(report)
        self.plot(data)

    def plot(self, data):
        fc = data['fc']
        fm = data['fm']
        runtime = data['runtime']

        # TEMPORAL -----------------------------------------------------------------------------------------------------
        xt = data['x']
        yt = data['y']
        mt = data['mt']
        fft_length = data['fft_length']

        try:
            x_periods = 4
            xt_left = 0
            xt_right = min((x_periods / fm), runtime)
        except ZeroDivisionError:
            raise ValueError("Carrier frequency must be non-zero!")

        ylimit = np.max(np.abs(yt)) * 1.25
        yt_tick = ylimit / 4

        # SPECTRAL -----------------------------------------------------------------------------------------------------
        xf = data['xf']
        yf = data['ywf']
        bw = data['bw']
        bw_margin = 3*fm

        Fs = data['Fs']
        N = data['N']
        yrms = data['RMS']

        freq_end = Fs * ((fft_length-1) / N)
        xf_left = max((fc - bw / 2) - bw_margin, 0)
        xf_right = min(fc + bw / 2 + bw_margin, freq_end)  # Does not exceed max bin

        dB = False
        if dB:
            # decibel
            yf_btm = -250
            yf_top = 0
            yf_tick = 50
            yf_ticks = np.arange(yf_btm, yf_top, yf_tick)
        else:
            # linear
            n_tick = 5
            yf_data_peak = round(abs(max(yf)), 1)
            yf_tick = round(np.ceil((yf_data_peak / (n_tick - 2)) * 10) / 10, 1)

            # https://stackoverflow.com/a/66805331/3382269
            yf_btm = -yf_tick
            yf_top = round((n_tick - 2) * yf_tick, 1)
            yf_ticks = np.arange(yf_btm, yf_top + yf_tick, yf_tick)


        bw_text = f"BW: {round(bw / 1000, 2)}kHz"
        dim_left = fc - bw / 2
        dim_right = fc + bw / 2
        dim_height = yf_btm / 2

        params = {'xt': xt * 1000, 'yt': yt, 'mt': mt,
                  'xt_left': xt_left, 'xt_right': 1e3 * xt_right,
                  'yt_btm': -ylimit, 'yt_top': ylimit + yt_tick, 'yt_tick': yt_tick,

                  'xf': xf[:fft_length] / 1000, 'yf': np.abs(yf[:fft_length]), 'bw': bw / 1000,
                  'xf_left': xf_left / 1000, 'xf_right': xf_right / 1000,
                  'yf_btm': yf_btm, 'yf_top': yf_top, 'yf_ticks': yf_ticks,
                  'dim_left': dim_left / 1000, 'dim_right': dim_right / 1000, 'dim_height': dim_height,
                  'bw_text': bw_text
                  }

        self.frame.plot(params)
