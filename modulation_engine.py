import os
import time
import datetime
from pathlib import Path

import csv

import numpy as np
import threading

import matplotlib.pyplot as plt  # only used for running this file directly


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

    :param f0: frequency of interest
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
    yt -= np.mean(yt)

    # Calculate windowing function and its length ----------------------------------------------------------------------
    if windfunc == 'bartlett':
        w = np.bartlett(N)
        main_lobe_width = 4 * (Fs / N)
    elif windfunc == 'hanning':
        w = np.hanning(N)
        main_lobe_width = 4 * (Fs / N)
    elif windfunc == 'hamming':
        w = np.hamming(N)
        main_lobe_width = 4 * (Fs / N)
    elif windfunc == 'blackman':
        w = np.blackman(N)
        main_lobe_width = 6 * (Fs / N)
    else:
        # TODO - maybe include kaiser as well, but main lobe width varies with alpha
        raise ValueError("Invalid windowing function selected!")

    # Calculate amplitude correction factor after windowing ------------------------------------------------------------
    # https://stackoverflow.com/q/47904399/3382269
    amplitude_correction_factor = 1 / np.mean(w)

    # Calculate the length of the FFT ----------------------------------------------------------------------------------
    if (N % 2) == 0:
        # for even values of N: FFT length is (N / 2) + 1
        fft_length = int(N / 2) + 1
    else:
        # for odd values of N: FFT length is (N + 1) / 2
        fft_length = int((N + 2) / 2)

    """
    Compute the FFT of the signal Divide by the length of the FFT to recover the original amplitude. Note dividing 
    alternatively by N samples of the time-series data splits the power between the positive and negative sides. 
    However, we are only looking at one side of the FFT.
    """
    yf_fft = (np.fft.fft(yt * w) / fft_length) * amplitude_correction_factor

    yf_rfft = yf_fft[:fft_length]
    xf_fft = np.linspace(0.0, Fs, N)
    xf_rfft = np.linspace(0.0, Fs / 2, fft_length)

    return xf_fft, yf_fft, xf_rfft, yf_rfft, fft_length, main_lobe_width


class Modulators:
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, parent):
        self.frame = parent
        self.DUMMY_DATA = False  # can be toggled by the gui
        self.user_values_good = False  # Flag indicates user input for amplitude value is good (True)

        self.params = {'mode': 0, 'source': 0, 'amplitude': '', 'units': '',
                       'rms': 0, 'frequency': 0.0, 'error': 0.0,
                       'cycles': 0.0, 'filter': ''}
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
        message = f"Modulation: {modulation_index} @ {message_frequency} Hz with {message_phase} (deg) phase shift"
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
        Ac = self.params['carrier_amplitude']
        fc = self.params['carrier_frequency']
        modulation_index = self.params['modulation_index']
        fm = self.params['message_frequency']
        message_phase = self.params['message_phase']

        # time base ----------------------------------------------------------------------------------------------------
        if fm != 0:
            N = getWindowLength(f0=fm, fs=sample_rate, windfunc=WINDOW_FUNC, error=main_lobe_error)
        elif fm == 0:
            ldf = fc / 10  # lowest detectable frequency
            N = getWindowLength(f0=ldf, fs=sample_rate, windfunc=WINDOW_FUNC, error=main_lobe_error)
        else:
            raise ValueError("Carrier frequency must be non-zero!")

        runtime = N / sample_rate
        N_range = np.arange(0, N, 1)
        xt = N_range / sample_rate

        # Compute message ==============================================================================================
        if waveform_type == 'sine':
            T = sample_rate / fm  # sample length per period

            # message signal
            mt = np.sin(2 * np.pi * fm * xt + np.deg2rad(message_phase))

            # integral of message signal
            mt_ = (1 / (np.pi * fm)) * np.sin(np.pi * fm * xt) * np.sin(np.pi * fm * xt + message_phase)
        # --------------------------------------------------------------------------------------------------------------
        elif waveform_type == 'triangle':
            # Triangle
            # https://en.wikipedia.org/wiki/Triangle_wave#Modulo_operation
            T = sample_rate / fm  # sample length per period

            # message signal (modulo operation)
            N_phase_shifted = N_range + int(T * message_phase / 360)

            mt = (4 / T) * np.abs(((N_phase_shifted - T / 4) % T) - T / 2) - 1

            # integral of triangle
            integral_shift = N_phase_shifted + T / 4

            def first(x, T):
                t = (x % T)
                return 2 / T * t ** 2 - t

            def second(x, T):
                t = (x % T)
                return -2 / T * t ** 2 + 3 * t - T

            scaling = 1 / sample_rate
            mt_ = scaling * np.where((integral_shift % T) < T / 2, first(integral_shift, T), second(integral_shift, T))

        # --------------------------------------------------------------------------------------------------------------
        elif waveform_type == 'sawtooth':
            # Sawtooth
            T = sample_rate / fm  # sample length per period
            N_phase_shifted = N_range + int(T * message_phase / 360)

            mt = (N_phase_shifted % T) / T

            # integral of sawtooth
            scaling = 1 / sample_rate
            mt_ = scaling * ((1/T) * (N_phase_shifted % T) ** 2 - (N_phase_shifted % T))

        # --------------------------------------------------------------------------------------------------------------
        elif waveform_type == 'square':
            T = sample_rate / fm
            N_phase_shifted = N_range + int(T * message_phase / 360)

            # message signal
            mt = np.where((N_phase_shifted % T) < T / 2, 1, 0)

            # integral of message signal (square wave integral is a triangle wave (modulo operation) with 180 phase lag)
            scaling = 1 / sample_rate
            mt_ = scaling * np.where((N_phase_shifted % T) < T / 2, (N_phase_shifted % T), -(N_phase_shifted % T))

            if modulation_type == 2:
                mt = np.deg2rad(message_phase * mt)

        # --------------------------------------------------------------------------------------------------------------
        elif waveform_type == 'keying':
            digital_modulation = {0: 'ask', 1: 'fsk', 2: 'psk'}
            T = sample_rate / fm

            # message signal -------------------------------------------------------------------------------------------
            binary_message = np.round(np.random.rand(1, int(N / T)))[0]
            mt = np.repeat(binary_message, T)[:N]

            # frequency term is converted to an angle. Since discrete steps, there are only f_low (0) and f_high (1)
            mt_ = (2 * mt - 1) * xt

            if digital_modulation[modulation_type] == 'psk':
                mt = np.deg2rad(message_phase * mt)

        # --------------------------------------------------------------------------------------------------------------
        else:
            raise ValueError("Invalid waveform type selected!")

        # waveform data ------------------------------------------------------------------------------------------------
        # https://user.eng.umd.edu/~tretter/commlab/c6713slides/ch8.pdf
        # --------------------------------------------------------------------------------------------------------------
        if modulation_type == 0:
            # amplitude modulation
            ct = Ac * np.sin(2 * np.pi * fc * xt)
            st = ct * modulation_index * mt
            bw = 2 * fm

        # --------------------------------------------------------------------------------------------------------------
        elif modulation_type == 1:
            # frequency modulation -------------------------------------------------------------------------------------
            # In FM, the angle is directly proportional to the integral of m(t)
            # https://math.stackexchange.com/q/178079
            # modulation index (mi) = (Am * kf) / fm
            # kf = (mi) * fm / Am
            st = Ac * np.sin(2 * np.pi * (fc * xt + (modulation_index * fm * mt_)))

            if waveform_type != 'keying':
                bw = 2 * fm * (modulation_index + 1)
            else:
                bw = 2 * (1 / T + modulation_index * fm) + fm

        # --------------------------------------------------------------------------------------------------------------
        elif modulation_type == 2:
            # phase modulation
            # In PM, the angle is directly proportional to m(t)
            st = Ac * np.sin(2 * np.pi * fc * xt + modulation_index * mt)

            if waveform_type != 'keying':
                bw = 2 * fm * (modulation_index + 1)
            else:
                bw = fc + fm - (fc - fm)

        # --------------------------------------------------------------------------------------------------------------
        else:
            raise ValueError("Invalid modulation type selected!")

        self.fft(xt, st, mt, runtime, bw, fc, fm, sample_rate, N, WINDOW_FUNC)

    def fft(self, xt, yt, mt, runtime, bw, fc, fm, Fs, N, WINDOW_FUNC='blackman'):
        yrms = rms_flat(yt)

        # FFT ==========================================================================================================
        xf_fft, yf_fft, xf_rfft, yf_rfft, fft_length, main_lobe_width = windowed_fft(yt, Fs, N, WINDOW_FUNC)

        data = {'x': xt, 'y': yt, 'bw': bw, 'mt': mt, 'xf': xf_rfft, 'ywf': yf_rfft, 'fft_length': fft_length,
                'RMS': yrms, 'N': N, 'runtime': runtime, 'Fs': Fs, 'fc': fc, 'fm': fm}

        # save measurement to csv --------------------------------------------------------------------------------------
        header = ['xt', 'yt', 'xf', 'yf']
        write_to_csv('results/history', 'modulation_schemes', header, xt, yt, xf_fft, yf_fft)

        # plot and report ----------------------------------------------------------------------------------------------
        report = {'yrms': yrms, 'bw': bw}
        try:
            self.frame.results_update(report)
        except AttributeError:
            print('Attempted to update results of a parent frame or panel. Skipped.')
            pass
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

        x_periods = 8
        xt_left = 0
        if fm != 0:
            xt_right = min((x_periods / fm), runtime)
        elif fm == 0:
            xt_right = min((x_periods / fc), runtime)
        else:
            raise ValueError("Carrier frequency must be non-zero!")

        ylimit = np.max(np.abs(yt)) * 1.25
        yt_tick = ylimit / 4

        # SPECTRAL -----------------------------------------------------------------------------------------------------
        xf = data['xf']
        yf = data['ywf']
        bw = data['bw']
        bw_margin = 2 * fm

        Fs = data['Fs']
        N = data['N']

        freq_end = Fs * ((fft_length - 1) / N)
        if fm != 0:
            xf_left = max((fc - bw / 2) - bw_margin, 0)
            xf_right = min(max(2 * fc - xf_left, fc + bw / 2), freq_end)  # Does not exceed max bin
        elif fm == 0:
            xf_left = max(fc - (fc / 2), 0)
            xf_right = min(2 * fc - fc / 2, freq_end)  # Does not exceed max bin
        else:
            raise ValueError("Carrier frequency must be non-zero!")

        dB = False
        if dB:
            # decibel
            yf_data_peak = round(max(abs(yf)), 1)
            yf_btm = -250
            yf_top = 0
            yf_tick = 50
            yf_ticks = np.arange(yf_btm, yf_top, yf_tick)
        else:
            # linear
            n_tick = 5
            yf_data_peak = round(max(abs(yf)), 1)
            yf_tick = round(np.ceil((yf_data_peak / (n_tick - 2)) * 10) / 10, 1)

            # https://stackoverflow.com/a/66805331/3382269
            yf_btm = -yf_tick
            yf_top = round((n_tick - 2) * yf_tick, 1)
            yf_ticks = np.arange(yf_btm, yf_top + yf_tick, yf_tick)

        bw_text = f"BW: {round(bw / 1000, 2)}kHz"
        dim_left = max(fc - bw / 2, 0)
        dim_right = min(fc + bw / 2, freq_end)
        dim_height = yf_btm / 2

        print("\nBoundaries:")
        print('xf_left:', xf_left, 'xf_right:', xf_right)
        print('yf_data_peak:', yf_data_peak, 'yf_btm', yf_btm, 'yf_top:', yf_top)
        print('dim_left:', dim_left, 'dim_right:', dim_right, 'dim_left:', dim_left)
        print()

        params = {'xt': xt * 1000, 'yt': yt, 'mt': mt * (1 / max(mt)),
                  'xt_left': xt_left, 'xt_right': 1e3 * xt_right,
                  'yt_btm': -ylimit, 'yt_top': ylimit + yt_tick, 'yt_tick': yt_tick,

                  'xf': xf[:fft_length] / 1000, 'yf': np.abs(yf[:fft_length]), 'bw': bw / 1000,
                  'xf_left': xf_left / 1000, 'xf_right': xf_right / 1000,
                  'yf_btm': yf_btm, 'yf_top': yf_top, 'yf_ticks': yf_ticks,
                  'dim_left': dim_left / 1000, 'dim_right': dim_right / 1000, 'dim_height': dim_height,
                  'bw_text': bw_text
                  }

        self.frame.plot(params)


class Test:
    def __init__(self):
        self.M = Modulators(parent=self)
        self.params = {'sample_rate': 80000,
                       'main_lobe_error': 0.06,  # error * 'freq of interest' becomes the 'lowest detectable freq'
                       'modulation_type': 1,  # Modulation type: 0 -> amplitude; 1 -> frequency; 2 -> phase
                       'waveform_type': 'keying',  # 'sine', 'triangle', 'square', 'keying'
                       'carrier_amplitude': 1,
                       'carrier_frequency': 1000,
                       'modulation_index': 5,
                       'message_frequency': 100,
                       'message_phase': 0  # for PSK, message phase becomes the phase shift of the carrier
                       }

        self.M.params = self.params
        self.M.simulation()

    def plot(self, params):
        # Plot objects -------------------------------------------------------------------------------------------------
        figure = plt.figure(figsize=(12.8, 4.8))  # standard figsize=(6.4, 4.8)

        ax1 = figure.add_subplot(211)
        ax2 = figure.add_subplot(212)

        temporal, = ax1.plot([], [], linestyle='-')
        temporal_2, = ax1.plot([], [], linestyle='--')
        spectral, = ax2.plot([], [], color='#C02942')

        # Plot Annotations ---------------------------------------------------------------------------------------------
        # https://stackoverflow.com/a/38677732
        arrow_dim_obj = ax2.annotate("", xy=(0, 0), xytext=(0, 0),
                                     textcoords=ax2.transData, arrowprops=dict(arrowstyle='<->'))
        bar_dim_obj = ax2.annotate("", xy=(0, 0), xytext=(0, 0),
                                   textcoords=ax2.transData, arrowprops=dict(arrowstyle='|-|'))
        bbox = dict(fc="white", ec="none")
        dim_text = ax2.text(0, 0, "", ha="center", va="center", bbox=bbox)

        # TEMPORAL -----------------------------------------------------------------------------------------------------
        xt = params['xt']
        yt = params['yt']
        mt = params['mt']

        temporal.set_data(xt, yt)
        temporal_2.set_data(xt, mt)

        xt_left = params['xt_left']
        xt_right = params['xt_right']
        yt_btm = params['yt_btm']
        yt_top = params['yt_top']
        yt_tick = params['yt_tick']

        ax1.set_xlim(left=xt_left, right=xt_right)
        # self.ax1.set_yticks(np.arange(yt_btm, yt_top, yt_tick))

        # SPECTRAL -----------------------------------------------------------------------------------------------------
        xf = params['xf']
        yf = params['yf']

        spectral.set_data(xf, yf)

        xf_left = params['xf_left']
        xf_right = params['xf_right']
        yf_btm = params['yf_btm']
        yf_top = params['yf_top']
        yf_ticks = params['yf_ticks']

        ax2.set_xlim(left=xf_left, right=xf_right)
        ax2.set_ylim(bottom=yf_btm, top=yf_top)
        ax2.set_yticks(yf_ticks)

        # Labels -------------------------------------------------------------------------------------------------------
        ax1.set_title('SAMPLED TIMED SERIES DATA')
        ax1.set_xlabel('TIME (ms)')
        ax1.set_ylabel('AMPLITUDE')

        ax2.set_title('SPECTRAL DATA')
        ax2.set_xlabel('FREQUENCY (kHz)')
        ax2.set_ylabel('AMPLITUDE (V)')
        ax2.grid()
        figure.align_ylabels([ax1, ax2])
        figure.tight_layout()

        # Annotations --------------------------------------------------------------------------------------------------
        dim_height = params['dim_height']
        dim_left = params['dim_left']
        dim_right = params['dim_right']
        bw = params['bw']

        # Arrow dimension line update ----------------------------------------------------------------------------------
        # https://stackoverflow.com/a/48684902 -------------------------------------------------------------------------
        arrow_dim_obj.xy = (dim_left, dim_height)
        arrow_dim_obj.set_position((dim_right, dim_height))
        arrow_dim_obj.textcoords = ax2.transData

        # Bar dimension line update ------------------------------------------------------------------------------------
        bar_dim_obj.xy = (dim_left, dim_height)
        bar_dim_obj.set_position((dim_right, dim_height))
        bar_dim_obj.textcoords = ax2.transData

        # dimension text update ----------------------------------------------------------------------------------------
        dim_text.set_position((dim_left + (bw / 2), dim_height))
        dim_text.set_text(params['bw_text'])

        # UPDATE PLOT FEATURES -----------------------------------------------------------------------------------------
        figure.tight_layout()
        ax1.relim()  # recompute the ax.dataLim
        ax1.margins(x=0)
        ax1.autoscale(axis='y')
        plt.show()


if __name__ == "__main__":
    T = Test()
