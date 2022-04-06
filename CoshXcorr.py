from .CoshConfig import CoshConfig
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import time


class CoshXcorr(object):
    def __init__(self, trace1=None, trace2=None, config: CoshConfig = CoshConfig()):
        self.config = config
        if trace1 is None:
            trace1 = [0, 0, 0]
        if trace2 is None:
            trace2 = trace1
        self.trace1 = trace1
        self.trace2 = trace2

        # Following data will be updated after calling method process()
        self.phasechange1 = None
        self.phasechange2 = None
        self.psd11 = None
        self.psd11_err = None
        self.psd22 = None
        self.psd22_err = None
        self.psd12 = None
        self.psd12_err = None
        # self.freq_list = None
        # self.freq_filter = None

    @property
    def freq_list(self):
        freq_list = np.array([])
        for x in self.config.offset_freq_list:
            freq_list = np.append(freq_list, x)
        return freq_list

    @property
    def freq_filter(self):
        freq_filter = np.array([])
        for x in self.config.offset_freq_filter_list:
            freq_filter = np.append(freq_filter, x)
        return freq_filter

    def process(self, hilbert=scipy.signal.hilbert, fft=np.fft.fft, print_progress=True):
        if print_progress:
            self.config.print_config()
        t_process_start = time.time()
        trace1 = np.array(self.trace1[self.config.range_start:self.config.range_stop])
        trace2 = np.array(self.trace2[self.config.range_start:self.config.range_stop])

        t_start = time.time()
        if print_progress:
            print("Calculating phase change using Hilbert Transformation...")
        phasechange1 = np.mod(np.diff(np.angle(hilbert(trace1 - np.mean(trace1)))), 2 * np.pi)
        phasechange2 = np.mod(np.diff(np.angle(hilbert(trace2 - np.mean(trace2)))), 2 * np.pi)
        self.phasechange1 = phasechange1
        self.phasechange2 = phasechange2
        if print_progress:
            print(f"Hilbert Transformation finished in {time.time() - t_start:.3f} second(s).")

        psd11 = np.array([])
        psd11_err = np.array([])
        psd22 = np.array([])
        psd22_err = np.array([])
        psd12 = np.array([])
        psd12_err = np.array([])
        for ii, bw in enumerate(self.config.bw_segment[:-1]):
            segment_length = int(np.round(1 / (bw * self.config.time_unit)))
            segment_count = int(np.floor(phasechange1.__len__() / segment_length))
            if print_progress:
                message = f"Calculating frequency range {self.config.bw_segment[ii] * self.config.offset_start_ratio} Hz " \
                          f"to {self.config.bw_segment[ii + 1] * self.config.offset_start_ratio} Hz " \
                          f"with bandwidth {self.config.bw_segment[ii]} Hz. " \
                          f"segment_length={segment_length}, segment_count={segment_count}..."
                print(message)

            t_start = time.time()
            offset_pos = self.config.offset_pos_list[ii]
            # offset_freq = self.config.offset_freq_list[ii] (Not necessary here)

            pc1seg = phasechange1[:segment_count * segment_length].reshape((segment_count, segment_length))
            pc2seg = phasechange2[:segment_count * segment_length].reshape((segment_count, segment_length))

            pc1f = fft(pc1seg) / segment_length
            pc2f = fft(pc2seg) / segment_length

            cor11 = pc1f[:, offset_pos] * np.conj(pc1f[:, offset_pos]) / np.power(2 * np.pi * self.config.time_unit,
                                                                                  2) / bw  # rms^2/BW
            cor22 = pc2f[:, offset_pos] * np.conj(pc2f[:, offset_pos]) / np.power(2 * np.pi * self.config.time_unit,
                                                                                  2) / bw
            cor12 = pc1f[:, offset_pos] * np.conj(pc2f[:, offset_pos]) / np.power(2 * np.pi * self.config.time_unit,
                                                                                  2) / bw
            psd11 = np.append(psd11, np.mean(cor11, axis=0))
            psd11_err = np.append(psd11_err, np.std(cor11, axis=0) / np.sqrt(segment_count))
            psd22 = np.append(psd22, np.mean(cor22, axis=0))
            psd22_err = np.append(psd22_err, np.std(cor22, axis=0) / np.sqrt(segment_count))
            psd12 = np.append(psd12, np.mean(cor12, axis=0))
            psd12_err = np.append(psd12_err, np.std(cor12, axis=0) / np.sqrt(segment_count))
            if print_progress:
                print(f"Segment process finished in {time.time() - t_start:.3f} second(s).")

        self.psd11 = psd11
        self.psd11_err = psd11_err
        self.psd22 = psd22
        self.psd22_err = psd22_err
        self.psd12 = psd12
        self.psd12_err = psd12_err
        print(f"All data processing finished in {time.time() - t_process_start:.3f} second(s).")

    def process_gpu(self, print_progress=True):
        print("Trying to process using CUDA GPU...")
        import torch
        if not torch.cuda.is_available():
            print("CUDA is not available. Should use normal process instead.")
            return
        self.process(hilbert=self.__hilbert_gpu, fft=self.__fft_gpu, print_progress=print_progress)

    def plot_SSB_freq_noise(self, freq_lim=None):
        if freq_lim is None:
            freq_lim = [np.min(self.freq_list), np.max(self.freq_list)]

        plt_index = np.logical_and(self.freq_list > np.min(freq_lim), self.freq_list < np.max(freq_lim))
        plt_freq = self.freq_list[plt_index]
        fn = np.abs(self.psd12 / self.freq_filter)[plt_index]
        fn_err = np.abs(self.psd12_err / self.freq_filter)[plt_index]
        plt.figure(figsize=(12, 6))
        plt.errorbar(plt_freq, fn, yerr=fn_err)
        plt.loglog()
        plt.fill_between(plt_freq, fn - fn_err, fn + fn_err, alpha=0.3)
        plt.xlabel('Frequency offset (Hz)')
        plt.ylabel('SSB frequency noise (Hz^2/Hz)')

    def plot_SSB_phase_noise(self, freq_lim=None):
        if freq_lim is None:
            freq_lim = [np.min(self.freq_list), np.max(self.freq_list)]

        plt_index = np.logical_and(self.freq_list > np.min(freq_lim), self.freq_list < np.max(freq_lim))
        plt_freq = self.freq_list[plt_index]
        fn = self.psd12 / self.freq_filter
        fn_err = self.psd12_err / self.freq_filter
        fpn = np.abs(fn / np.power(self.freq_list, 2))[plt_index]
        fpn_err = np.abs(fn_err / np.power(self.freq_list, 2))[plt_index]
        plt.figure(figsize=(12, 6))
        plt.errorbar(plt_freq, fpn, yerr=fpn_err)
        plt.loglog()
        plt.fill_between(plt_freq, fpn - fpn_err, fpn + fpn_err, alpha=0.3)
        plt.xlabel('Frequency offset (Hz)')
        plt.ylabel('SSB phase noise (rad^2/Hz)')

    def __hilbert_gpu(self, data):
        import torch
        N = data.shape[-1]
        # Allocates memory on GPU with size/dimensions of signal
        data_gpu = torch.tensor(data, dtype=torch.float32).cuda()
        transforms = torch.fft.fft(data_gpu, axis=-1)

        del data_gpu
        torch.cuda.empty_cache()

        transforms[1:N // 2] *= 2  # positive frequency
        transforms[N // 2:] *= 0  # negative frequency
        # Do not change DC signal
        result = torch.fft.ifft(transforms).cpu()

        del transforms
        torch.cuda.empty_cache()

        return result.numpy()

    def __fft_gpu(self, data):
        import torch
        data_gpu = torch.tensor(data, dtype=torch.float32).cuda()
        transforms = torch.fft.fft(data_gpu, axis=-1).cpu()
        del data_gpu
        torch.cuda.empty_cache()
        return transforms.numpy()
