import numpy as np


class CoshConfig(object):
    def __init__(self, delay_freq=214.06e3,
                 bw_segment=None, sample_rate=250e6,
                 offset_start_ratio: int = 20,
                 range_start: int = None, range_stop: int = None):
        if bw_segment is None:
            bw_segment = [5e2, 20e2, 5e3, 20e3, 50e3, 200e3, 1000e3]
        self.delay_freq = delay_freq
        self.bw_segment = bw_segment
        self.sample_rate = sample_rate

        self.offset_start_ratio = offset_start_ratio  # freq_start = offset_start_ratio*bw_segment[0]

        self.range_start = int(range_start)
        self.range_stop = int(range_stop)

    @property
    def time_unit(self):
        return 1 / self.sample_rate

    @property
    def offset_pos_list(self):
        return [range(self.offset_start_ratio,
                      int(np.round(self.offset_start_ratio * self.bw_segment[ii + 1] / self.bw_segment[ii])))
                for ii in range(self.bw_segment.__len__() - 1)]

    @property
    def offset_freq_list(self):
        return [np.array(self.offset_pos_list[ii])*self.bw_segment[ii] for ii in range(self.bw_segment.__len__() - 1)]

    @property
    def offset_freq_filter_list(self):
        return [2 + (np.power(2 * np.sin(np.pi * self.offset_freq_list[ii] / self.delay_freq), 2) - 2)
                * np.max([1 - self.bw_segment[ii] / self.delay_freq, 0])
                # modulation to PSD caused by delay line
                # sinc^2 filter is applied, net effect for sine signal is to reduce fringes contrast
                for ii in range(self.bw_segment.__len__() - 1)]

    def print_config(self):
        message = "Cosh configuration summary".center(80, '-') + "\n"
        message = message + f"|\tUsed data index range: {self.range_start} to {self.range_stop}. " \
                            f"\t(Change by self.config.range_start = 0, self.config.range_stop = None)\n"
        message = message + f"|\tDelay line FSR = {self.delay_freq} Hz. " \
                            f"\t(Change by self.config.delay_freq=100e6)\n"
        message = message + f"|\tTrace sampling rate = {self.sample_rate} Hz. " \
                            f"\t(Change by self.config.sample_rate=250e6)\n"
        message = message + f"|\tBandwidth segment = {self.bw_segment}. \n" \
                            f"|\t\t(Change by self.config.bw_segment=[5e2, 20e2])\n"
        message = message + f"|\toffset start ratio = {self.offset_start_ratio}. " \
                            f"\t(Change by self.config.offset_start_ratio=10)\n"
        message = message + f"|\tFrequency windows = offset_start_ratio * bw_segment\n" \
                            f"|\t                  = {[self.offset_start_ratio*x for x in self.bw_segment]}\n"
        message = message + "Cosh configuration summary Ends".center(80, '-')
        print(message)
