from phase_modulation import Modulators as pm

import numpy as np
import threading

import wx

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar

# https://stackoverflow.com/a/38251497
# https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html
pylab_params = {'legend.fontsize': 'medium',
                'font.family': 'Segoe UI',
                'axes.titleweight': 'bold',
                'figure.figsize': (15, 5),
                'axes.labelsize': 'medium',
                'axes.titlesize': 'medium',
                'xtick.labelsize': 'medium',
                'ytick.labelsize': 'medium'}
pylab.rcParams.update(pylab_params)


# ======================================================================================================================
def to_float(str_val, property=None):
    try:
        return float(str_val)
    except ValueError:
        if property:
            raise ValueError(f"{str_val} could not be converted to float for the property: {property}!")
        else:
            raise ValueError(f"{str_val} could not be converted to float!")


# ======================================================================================================================
class PhaseModulatorPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, wx.ID_ANY)

        self.parent = parent

        self.left_panel = wx.Panel(self, wx.ID_ANY)
        self.plot_panel = wx.Panel(self, wx.ID_ANY, style=wx.SIMPLE_BORDER)

        # PANELS =======================================================================================================
        # LEFT Panel ---------------------------------------------------------------------------------------------------
        self.text_sample_rate = wx.TextCtrl(self.left_panel, wx.ID_ANY, "80000")
        self.text_mainlobe_error = wx.TextCtrl(self.left_panel, wx.ID_ANY, "0.1")

        self.text_carrier_amplitude = wx.TextCtrl(self.left_panel, wx.ID_ANY, "1")
        self.text_carrier_frequency = wx.TextCtrl(self.left_panel, wx.ID_ANY, "1e3")

        self.combo_waveform = wx.ComboBox(self.left_panel, wx.ID_ANY,
                                          choices=["Sine", "Triangle", "Square"],
                                          style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.combo_modulation = wx.ComboBox(self.left_panel, wx.ID_ANY,
                                            choices=["Amplitude", "Frequency", "Phase"],
                                            style=wx.CB_DROPDOWN | wx.CB_READONLY)

        self.text_modulation_index = wx.TextCtrl(self.left_panel, wx.ID_ANY, "1")
        self.text_message_frequency = wx.TextCtrl(self.left_panel, wx.ID_ANY, "100")
        self.text_message_phase = wx.TextCtrl(self.left_panel, wx.ID_ANY, "0")
        self.text_report_rms = wx.TextCtrl(self.left_panel, wx.ID_ANY, "", style=wx.TE_READONLY)
        self.text_report_bw = wx.TextCtrl(self.left_panel, wx.ID_ANY, "", style=wx.TE_READONLY)

        self.btn_start = wx.Button(self.left_panel, wx.ID_ANY, "RUN")
        self.combo_mode = wx.ComboBox(self.left_panel, wx.ID_ANY,
                                      choices=["Single", "Continuous"],
                                      style=wx.CB_DROPDOWN)

        # PLOT Panel ---------------------------------------------------------------------------------------------------
        self.figure = plt.figure(figsize=(1, 1))  # look into Figure((5, 4), 75)
        self.canvas = FigureCanvas(self.plot_panel, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        # instance variables -------------------------------------------------------------------------------------------
        self.flag_complete = True  # Flag indicates any active threads (False) or thread completed (True)
        self.t = threading.Thread()
        self.pm = pm(self)
        self.user_input = {}

        # Plot objects -------------------------------------------------------------------------------------------------
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)

        self.temporal, = self.ax1.plot([], [], linestyle='-')
        self.temporal_2, = self.ax1.plot([], [], linestyle='--')
        self.spectral, = self.ax2.plot([], [], color='#C02942')

        # Plot Annotations ---------------------------------------------------------------------------------------------
        # https://stackoverflow.com/a/38677732
        self.arrow_dim_obj = self.ax2.annotate("", xy=(0, 0), xytext=(0, 0),
                                               textcoords=self.ax2.transData, arrowprops=dict(arrowstyle='<->'))
        self.bar_dim_obj = self.ax2.annotate("", xy=(0, 0), xytext=(0, 0),
                                             textcoords=self.ax2.transData, arrowprops=dict(arrowstyle='|-|'))
        bbox = dict(fc="white", ec="none")
        self.dim_text = self.ax2.text(0, 0, "", ha="center", va="center", bbox=bbox)

        # BINDINGS =====================================================================================================
        # Run Measurement (start subprocess) ---------------------------------------------------------------------------
        on_run_event = lambda event: self.on_run(event)
        self.Bind(wx.EVT_BUTTON, on_run_event, self.btn_start)

        on_combo_modulation_select = lambda event: self.combo_modulation_select(event)
        self.Bind(wx.EVT_COMBOBOX_CLOSEUP, on_combo_modulation_select, self.combo_modulation)

        self.__set_properties()
        self.__do_layout()
        self.__do_plot_layout()

    def __set_properties(self):
        self.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.canvas.SetMinSize((700, 490))

        self.left_panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.plot_panel.SetBackgroundColour(wx.Colour(255, 255, 255))

        self.left_panel.SetMinSize((310, 502))
        # self.left_sub_panel.SetBackgroundColour(wx.Colour(255, 0, 255))
        self.plot_panel.SetMinSize((700, 502))
        self.combo_modulation.SetSelection(0)
        self.combo_waveform.SetSelection(0)
        self.canvas.SetMinSize((700, 490))

        self.combo_mode.SetSelection(0)
        self.combo_mode.SetMinSize((110, 23))

    def __do_layout(self):
        sizer_2 = wx.GridSizer(1, 1, 0, 0)
        grid_sizer_1 = wx.FlexGridSizer(1, 2, 0, 0)
        grid_sizer_left_panel = wx.GridBagSizer(0, 0)
        grid_sizer_left_sub_btn_row = wx.GridBagSizer(0, 0)
        grid_sizer_plot = wx.GridBagSizer(0, 0)

        # LEFT PANEL ===================================================================================================
        # TITLE --------------------------------------------------------------------------------------------------------
        label_1 = wx.StaticText(self.left_panel, wx.ID_ANY, "MODULATION SCHEMES")
        label_1.SetFont(wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        grid_sizer_left_panel.Add(label_1, (0, 0), (1, 2), 0, 0)

        # static_line_1 = wx.StaticLine(self.left_panel, wx.ID_ANY)
        # static_line_1.SetMinSize((300, 2))
        # grid_sizer_left_panel.Add(static_line_1, (1, 0), (1, 3), wx.BOTTOM | wx.RIGHT | wx.TOP, 5)

        # SIMULATION SETUP ---------------------------------------------------------------------------------------------
        label_source = wx.StaticText(self.left_panel, wx.ID_ANY, "Simulation Settings")
        label_source.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        grid_sizer_left_panel.Add(label_source, (2, 0), (1, 2), wx.TOP, 10)

        static_line_2 = wx.StaticLine(self.left_panel, wx.ID_ANY)
        static_line_2.SetMinSize((300, 2))
        grid_sizer_left_panel.Add(static_line_2, (3, 0), (1, 3), wx.BOTTOM | wx.RIGHT | wx.TOP, 5)

        label_sample_rate = wx.StaticText(self.left_panel, wx.ID_ANY, "Sample Rate")
        grid_sizer_left_panel.Add(label_sample_rate, (4, 0), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, 5)
        grid_sizer_left_panel.Add(self.text_sample_rate, (4, 1), (1, 1), wx.BOTTOM | wx.LEFT, 5)
        label_Hz = wx.StaticText(self.left_panel, wx.ID_ANY, "(Hz)")
        grid_sizer_left_panel.Add(label_Hz, (4, 2), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)

        label_error = wx.StaticText(self.left_panel, wx.ID_ANY, "Error")
        grid_sizer_left_panel.Add(label_error, (5, 0), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, 5)
        grid_sizer_left_panel.Add(self.text_mainlobe_error, (5, 1), (1, 1), wx.BOTTOM | wx.LEFT, 5)
        label_percent = wx.StaticText(self.left_panel, wx.ID_ANY, "(%)")
        grid_sizer_left_panel.Add(label_percent, (5, 2), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)

        # CARRIER SETUP ------------------------------------------------------------------------------------------------
        label_source = wx.StaticText(self.left_panel, wx.ID_ANY, "Carrier Signal")
        label_source.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        grid_sizer_left_panel.Add(label_source, (6, 0), (1, 1), wx.TOP, 10)

        static_line_2 = wx.StaticLine(self.left_panel, wx.ID_ANY)
        static_line_2.SetMinSize((300, 2))
        grid_sizer_left_panel.Add(static_line_2, (7, 0), (1, 3), wx.BOTTOM | wx.RIGHT | wx.TOP, 5)

        label_carrier_amplitude = wx.StaticText(self.left_panel, wx.ID_ANY, "Amplitude:")
        grid_sizer_left_panel.Add(label_carrier_amplitude, (8, 0), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, 5)
        grid_sizer_left_panel.Add(self.text_carrier_amplitude, (8, 1), (1, 1), wx.BOTTOM | wx.LEFT, 5)

        label_carrier_frequency = wx.StaticText(self.left_panel, wx.ID_ANY, "Frequency:")
        grid_sizer_left_panel.Add(label_carrier_frequency, (9, 0), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, 5)
        grid_sizer_left_panel.Add(self.text_carrier_frequency, (9, 1), (1, 1), wx.LEFT, 5)
        label_Hz = wx.StaticText(self.left_panel, wx.ID_ANY, "(Hz)")
        grid_sizer_left_panel.Add(label_Hz, (9, 2), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)

        # MODULATION SETUP ---------------------------------------------------------------------------------------------
        label_source = wx.StaticText(self.left_panel, wx.ID_ANY, "Message Signal")
        label_source.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        grid_sizer_left_panel.Add(label_source, (10, 0), (1, 2), wx.TOP, 10)

        static_line_2 = wx.StaticLine(self.left_panel, wx.ID_ANY)
        static_line_2.SetMinSize((300, 2))
        grid_sizer_left_panel.Add(static_line_2, (11, 0), (1, 3), wx.BOTTOM | wx.RIGHT | wx.TOP, 5)

        label_modulation = wx.StaticText(self.left_panel, wx.ID_ANY, "Modulation:")
        grid_sizer_left_panel.Add(label_modulation, (12, 0), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, 5)
        grid_sizer_left_panel.Add(self.combo_modulation, (12, 1), (1, 1), wx.EXPAND | wx.BOTTOM | wx.LEFT, 5)

        label_waveform = wx.StaticText(self.left_panel, wx.ID_ANY, "Waveform:")
        grid_sizer_left_panel.Add(label_waveform, (13, 0), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, 5)
        grid_sizer_left_panel.Add(self.combo_waveform, (13, 1), (1, 1), wx.EXPAND | wx.BOTTOM | wx.LEFT, 5)

        label_modulation_index = wx.StaticText(self.left_panel, wx.ID_ANY, "Modulation Index:")
        grid_sizer_left_panel.Add(label_modulation_index, (14, 0), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, 5)
        grid_sizer_left_panel.Add(self.text_modulation_index, (14, 1), (1, 1), wx.BOTTOM | wx.LEFT, 5)

        label_message_frequency = wx.StaticText(self.left_panel, wx.ID_ANY, "Frequency:")
        grid_sizer_left_panel.Add(label_message_frequency, (15, 0), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, 5)
        grid_sizer_left_panel.Add(self.text_message_frequency, (15, 1), (1, 1), wx.BOTTOM | wx.LEFT, 5)
        label_Hz = wx.StaticText(self.left_panel, wx.ID_ANY, "(Hz)")
        grid_sizer_left_panel.Add(label_Hz, (15, 2), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM | wx.LEFT, 5)

        label_message_phase = wx.StaticText(self.left_panel, wx.ID_ANY, "Phase:")
        grid_sizer_left_panel.Add(label_message_phase, (16, 0), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, 5)
        grid_sizer_left_panel.Add(self.text_message_phase, (16, 1), (1, 1), wx.LEFT, 5)
        label_deg = wx.StaticText(self.left_panel, wx.ID_ANY, "(deg)")
        grid_sizer_left_panel.Add(label_deg, (16, 2), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)

        # METRICS (RESULTS) --------------------------------------------------------------------------------------------
        label_source = wx.StaticText(self.left_panel, wx.ID_ANY, "Metrics")
        label_source.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, 0, ""))
        grid_sizer_left_panel.Add(label_source, (17, 0), (1, 1), wx.TOP, 10)

        static_line_2 = wx.StaticLine(self.left_panel, wx.ID_ANY)
        static_line_2.SetMinSize((300, 2))
        grid_sizer_left_panel.Add(static_line_2, (18, 0), (1, 3), wx.BOTTOM | wx.RIGHT | wx.TOP, 5)

        label_report_rms = wx.StaticText(self.left_panel, wx.ID_ANY, "RMS:")
        grid_sizer_left_panel.Add(label_report_rms, (19, 0), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, 5)
        grid_sizer_left_panel.Add(self.text_report_rms, (19, 1), (1, 1), wx.BOTTOM | wx.LEFT, 5)

        label_report_bw = wx.StaticText(self.left_panel, wx.ID_ANY, "BW:")
        grid_sizer_left_panel.Add(label_report_bw, (20, 0), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, 5)
        grid_sizer_left_panel.Add(self.text_report_bw, (20, 1), (1, 1), wx.BOTTOM | wx.LEFT, 5)

        # BUTTONS ------------------------------------------------------------------------------------------------------
        static_line_4 = wx.StaticLine(self.left_panel, wx.ID_ANY)
        static_line_4.SetMinSize((300, 2))
        grid_sizer_left_panel.Add(static_line_4, (21, 0), (1, 3), wx.BOTTOM | wx.RIGHT | wx.TOP, 5)

        grid_sizer_left_sub_btn_row.Add(self.btn_start, (0, 0), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        grid_sizer_left_sub_btn_row.Add(self.combo_mode, (0, 1), (1, 1), wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        grid_sizer_left_panel.Add(grid_sizer_left_sub_btn_row, (22, 0), (1, 3), wx.ALIGN_TOP | wx.BOTTOM, 13)

        self.left_panel.SetSizer(grid_sizer_left_panel)

        # PLOT PANEL ===================================================================================================
        grid_sizer_plot.Add(self.canvas, (0, 0), (1, 1), wx.ALL | wx.EXPAND)
        grid_sizer_plot.Add(self.toolbar, (1, 0), (1, 1), wx.ALL | wx.EXPAND)
        grid_sizer_plot.AddGrowableRow(0)
        grid_sizer_plot.AddGrowableCol(0)
        self.plot_panel.SetSizer(grid_sizer_plot)

        # add to main panel --------------------------------------------------------------------------------------------
        grid_sizer_1.Add(self.left_panel, 0, wx.EXPAND | wx.RIGHT, 5)
        grid_sizer_1.Add(self.plot_panel, 1, wx.EXPAND, 5)
        grid_sizer_1.AddGrowableRow(0)
        grid_sizer_1.AddGrowableCol(1)

        sizer_2.Add(grid_sizer_1, 0, wx.EXPAND, 0)

        self.SetSizer(sizer_2)
        self.Layout()

    # ------------------------------------------------------------------------------------------------------------------
    def toggle_controls(self):
        if self.text_carrier_amplitude.Enabled:
            self.combo_modulation.Disable()
            self.text_sample_rate.Disable()
            self.text_mainlobe_error.Disable()
            self.text_carrier_amplitude.Disable()
            self.text_carrier_frequency.Disable()
            self.text_modulation_index.Disable()
            self.text_message_frequency.Disable()
            self.text_message_phase.Disable()

        else:
            self.combo_modulation.Enable()
            self.text_sample_rate.Enable()
            self.text_mainlobe_error.Enable()
            self.text_carrier_amplitude.Enable()
            self.text_carrier_frequency.Enable()
            self.text_modulation_index.Enable()
            self.text_message_frequency.Enable()
            self.text_message_phase.Enable()

    def combo_modulation_select(self, evt):
        if self.combo_modulation.GetSelection() == 0:
            # amplitude modulation
            self.text_modulation_index.SetValue('1')

        elif self.combo_modulation.GetSelection() == 1:
            # frequency modulation
            self.text_modulation_index.SetValue('5')

        elif self.combo_modulation.GetSelection() == 2:
            # phase modulation
            self.text_modulation_index.SetValue('5')

        else:
            raise ValueError("Invalid modulation selected!")

    # ------------------------------------------------------------------------------------------------------------------
    def get_values(self):
        mode = self.combo_mode.GetSelection()

        sample_rate = to_float(self.text_sample_rate.GetValue(), property="sample rate")
        main_lobe_error = to_float(self.text_mainlobe_error.GetValue(), property="main lobe error")
        modulation_type = self.combo_modulation.GetSelection()
        waveform_lookup = {0: 'sine', 1: 'triangle', 2: 'square'}
        waveform_type = waveform_lookup[self.combo_waveform.GetSelection()]
        carrier_amplitude = to_float(self.text_carrier_amplitude.GetValue(), property="carrier amplitude")
        carrier_frequency = to_float(self.text_carrier_frequency.GetValue(), property="carrier frequency")
        modulation_index = to_float(self.text_modulation_index.GetValue(), property="modulation index")
        message_frequency = to_float(self.text_message_frequency.GetValue(), property="message frequency")
        message_phase = to_float(self.text_message_phase.GetValue(), property="message phase")

        self.user_input = {'mode': mode,
                           'sample_rate': sample_rate,
                           'main_lobe_error': main_lobe_error,
                           'waveform_type': waveform_type,
                           'modulation_type': modulation_type,
                           'carrier_amplitude': carrier_amplitude,
                           'carrier_frequency': carrier_frequency,
                           'modulation_index': modulation_index,
                           'message_frequency': message_frequency,
                           'message_phase': message_phase,
                           }

        self.pm.user_values_good = True

    # ------------------------------------------------------------------------------------------------------------------
    def thread_this(self, func, arg=()):
        self.t = threading.Thread(target=func, args=arg, daemon=True)
        self.t.start()

    # ------------------------------------------------------------------------------------------------------------------
    def on_run(self, evt):
        self.get_values()
        if not self.t.is_alive() and self.flag_complete:
            # start new thread
            self.thread_this(self.pm.start, (self.user_input,))
            self.btn_start.SetLabel('STOP')

        elif self.t.is_alive() and self.user_input['mode'] == 1:
            # stop continuous
            # https://stackoverflow.com/a/36499538
            self.t.do_run = False
            self.btn_start.SetLabel('RUN')
        else:
            print('thread already running.')

    # ------------------------------------------------------------------------------------------------------------------
    def __do_plot_layout(self):
        self.ax1.set_title('SAMPLED TIMED SERIES DATA')
        self.ax1.set_xlabel('TIME (ms)')
        self.ax1.set_ylabel('AMPLITUDE')

        self.ax2.set_title('SPECTRAL DATA')
        self.ax2.set_xlabel('FREQUENCY (kHz)')
        self.ax2.set_ylabel('MAGNITUDE (dB)')
        self.ax2.grid()
        self.figure.align_ylabels([self.ax1, self.ax2])
        self.figure.tight_layout()

    def plot(self, params):
        # TEMPORAL -----------------------------------------------------------------------------------------------------
        xt = params['xt']
        yt = params['yt']
        mt = params['mt']

        self.temporal.set_data(xt, yt)
        self.temporal_2.set_data(xt, mt)

        xt_left = params['xt_left']
        xt_right = params['xt_right']
        yt_btm = params['yt_btm']
        yt_top = params['yt_top']
        yt_tick = params['yt_tick']

        self.ax1.set_xlim(left=xt_left, right=xt_right)
        # self.ax1.set_yticks(np.arange(yt_btm, yt_top, yt_tick))

        # SPECTRAL -----------------------------------------------------------------------------------------------------
        xf = params['xf']
        yf = params['yf']

        self.spectral.set_data(xf, yf)

        xf_left = params['xf_left']
        xf_right = params['xf_right']
        yf_btm = params['yf_btm']
        yf_top = params['yf_top']
        yf_ticks = params['yf_ticks']

        self.ax2.set_xlim(left=xf_left, right=xf_right)
        self.ax2.set_ylim(bottom=yf_btm, top=yf_top)
        self.ax2.set_yticks(yf_ticks)

        # Annotations --------------------------------------------------------------------------------------------------
        dim_height = params['dim_height']
        dim_left = params['dim_left']
        dim_right = params['dim_right']
        bw = params['bw']

        # Arrow dimension line update ----------------------------------------------------------------------------------
        # https://stackoverflow.com/a/48684902 -------------------------------------------------------------------------
        self.arrow_dim_obj.xy = (dim_left, dim_height)
        self.arrow_dim_obj.set_position((dim_right, dim_height))
        self.arrow_dim_obj.textcoords = self.ax2.transData

        # Bar dimension line update ------------------------------------------------------------------------------------
        self.bar_dim_obj.xy = (dim_left, dim_height)
        self.bar_dim_obj.set_position((dim_right, dim_height))
        self.bar_dim_obj.textcoords = self.ax2.transData

        # dimension text update ----------------------------------------------------------------------------------------
        self.dim_text.set_position((dim_left + (bw / 2), dim_height))
        self.dim_text.set_text(params['bw_text'])

        # REDRAW PLOT --------------------------------------------------------------------------------------------------
        self.plot_redraw()

    def plot_redraw(self):
        try:
            self.ax1.relim()  # recompute the ax.dataLim
        except ValueError:
            xt_length = len(self.ax1.get_xdata())
            yt_length = len(self.ax1.get_ydata())
            print(f'Are the lengths of xt: {xt_length} and yt: {yt_length} mismatched?')
            raise
        self.ax1.margins(x=0)
        self.ax1.autoscale(axis='y')

        # UPDATE PLOT FEATURES -----------------------------------------------------------------------------------------
        self.figure.tight_layout()

        self.toolbar.update()  # Not sure why this is needed - ADS
        self.canvas.draw()
        self.canvas.flush_events()

    def results_update(self, results):
        yrms = results['yrms']
        bw = results['bw']

        self.text_report_rms.SetValue(f"{'{:0.3e}'.format(yrms)}")
        self.text_report_bw.SetValue(f"{'{:0.3e}'.format(bw)}")

    def error_dialog(self, error_message):
        print(error_message)
        dial = wx.MessageDialog(None, str(error_message), 'Error', wx.OK | wx.ICON_ERROR)
        dial.ShowModal()


# FOR RUNNING INDEPENDENTLY ============================================================================================
class MyPhaseModulationFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        kwds["style"] = kwds.get("style", 0) | wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.SetSize((1055, 600))
        self.panel_frame = wx.Panel(self, wx.ID_ANY)
        self.panel_app = PhaseModulatorPanel(self.panel_frame)

        self.__set_properties()
        self.__do_layout()

    def __set_properties(self):
        self.SetTitle("Multimeter")
        self.panel_frame.SetBackgroundColour((255, 255, 255))

    def __do_layout(self):
        sizer_frame = wx.BoxSizer(wx.VERTICAL)
        sizer_frame_panel = wx.BoxSizer(wx.VERTICAL)

        sizer_frame.Add(self.panel_frame, 1, wx.EXPAND | wx.ALL, 0)
        self.SetSizer(sizer_frame)

        sizer_frame_panel.Add(self.panel_app, 1, wx.EXPAND | wx.ALL, 10)
        self.panel_frame.SetSizer(sizer_frame_panel)

        self.Layout()


class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyPhaseModulationFrame(None, wx.ID_ANY, "")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True


if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
