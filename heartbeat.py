import numpy as np
import math
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
from collections import defaultdict

def split_rrs(rr_intervals):
    """ Splits up all RR intervals into separate 1 minute windows

        :returns:
            Dictionary of RR interval windows
    """
    rr_split = defaultdict(list)
    counter = 0
    sum = 0
    for rr in rr_intervals:
        sum += rr
        rr_split[counter].append(rr)
        if (sum > 60000):
            counter += 1
            sum = 0
    return rr_split


def calc_rr_differences(rr_intervals):
    """ Calculates the RR differences and RR squared differences using the
        RR intervals

        :returns:
            The calculated RR differences and squared differences in two lists
    """
    rr_diffs = []
    rr_sqdiffs = []
    cnt = 0
    while (cnt < (len(rr_intervals)-1)):
        rr_diffs.append(abs(rr_intervals[cnt] - rr_intervals[cnt+1]))
        rr_sqdiffs.append(math.pow(rr_intervals[cnt] - rr_intervals[cnt+1], 2))
        cnt += 1
    return rr_diffs, rr_sqdiffs

def calc_time_domain_features(rr_intervals, rr_diffs, rr_sqdiffs, normalising=True):
    """ Calculates the time domain features using RR intervals, RR differences
        and RR squared differences

        Normalising will remove mathematical bias where HR can influence
        HRV values, demonstrated by Sacha et. al., divide by mean of RR intervals

        :returns:
            List of all time domain features:
                - Beats per minute (bpm)
                - Inter-beat interval (ibi)
                - Standard deviation of normal to normal RR intervals (SDNN)
                - Standard deviation of differences between adjacent NN
                  intervals (SDSD)
                - Root mean square of the successive differences (rMSSD)
                - The number of pairs of successive NNs that differ by more
                  than 20 ms (NN20)
                - The number of pairs of successive NNs that differ by more
                  than 50 ms (NN50)
                - The proportion of NN20 divided by total number of NNs (pNN20)
                - The proportion of NN50 divided by total number of NNs (pNN50)
    """
    td_features = {}
    td_features['bpm'] = 60000 / np.mean(rr_intervals)
    td_features['ibi'] = np.mean(rr_intervals)
    td_features['sdnn'] = np.std(rr_intervals)
    td_features['sdsd'] = np.std(rr_diffs)
    td_features['rmssd'] = np.sqrt(np.mean(rr_sqdiffs))
    NN20 = [x for x in rr_diffs if (x>20)]
    NN50 = [x for x in rr_diffs if (x>50)]
    td_features['nn20'] = NN20
    td_features['nn50'] = NN50
    td_features['pnn20'] = float(len(NN20)) / float(len(rr_diffs))
    td_features['pnn50'] = float(len(NN50)) / float(len(rr_diffs))

    if normalising:
        td_features['sdnn'] = td_features['sdnn'] / (np.mean(rr_intervals) / 1000)
        td_features['sdsd'] = td_features['sdsd'] / (np.mean(rr_intervals) / 1000)
        td_features['rmssd'] = td_features['rmssd'] / (np.mean(rr_intervals) / 1000)

    # plot_tachogram(rr_intervals)
    return td_features

def calc_timescale(rr_intervals):
    """ Calculates the timescale using the RR intervals e.g. 1000ms interval
        + 1020ms interval = timescale of [1000, 2020] ms

        :returns:
            The calculated timescale as a list of values in seconds
    """
    # Turn into seconds from ms
    time = np.cumsum([x / 1000 for x in rr_intervals])

    # Start at 0? time - time[0]
    return time - time[0]

def plot_tachogram(rr_intervals):
    """ Plots tachogram of RR intervals

        :returns:
            Plot
    """
    timescale = calc_timescale(rr_intervals)
    plt.title("RR Interval Tachogram")
    plt.plot(timescale, rr_intervals, label="RR Intervals", color='blue')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("RR Interval (ms)")
    # Draw plot so that it doesn't block computation (show needs to follow at end of computation)
    plt.draw()

def plot_hr_signal(hrs):
    """ Plots HR signal

        :returns:
            Plot
    """
    timescale = np.linspace(0, len(hrs), len(hrs))
    plt.title("HR Signal")
    plt.plot(timescale, hrs, label="HR", color='blue')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("HR (bpm)")
    # Draw plot so that it doesn't block computation (show needs to follow at end of computation)
    plt.draw()

def plot_interpolation(x, y, even_x, f):
    """ Plots original and interpolated signal

        :returns:
            Plot
    """
    plt.title("Original and Interpolated Signal")
    plt.plot(x, y, label="Original", color='blue')
    plt.plot(even_x, f(even_x), label="Interpolated", color='red')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("RR Interval (ms)")
    # Draw plot so that it doesn't block computation (show needs to follow at end of computation)
    plt.draw()

def plot_frequency_spectrum(frq, Y):
    """ Plots Frequency Spectrum of a HRV Window

        :returns:
            Plot
    """
    plt.title("Power Spectral Density of RR-Interval Window")
    plt.xlim(0,0.6) # Limit X axis to frequencies of interest (0-0.6Hz for visibility, we are interested in 0.04-0.5)
    # plt.ylim(0, 50) # Limit Y axis for visibility
    plt.plot(frq, abs(Y), c = 'blue')
    plt.xlabel(r'Frequency $(Hz)$')
    plt.ylabel(r'PSD $(s^2/Hz$)')

    # Draw plot so that it doesn't block computation (show needs to follow at end of computation)
    plt.draw()

def calc_frequency_domain_features(rr_intervals, fs, PLOT=True, normalising=True):
    """ Calculates the frequency domain features using RR intervals and a sampling rate -
        interpolates the signal (re-sampling), transforms the signal to the power spectrum (PSD)
        and integrates the area under the LF and HF portion of the spectrum

        Normalising will remove mathematical bias where HR can influence
        HRV values, demonstrated by Sacha et. al., divide by mean of RR
        intervals squared

        :returns:
            List of all frequency domain features:
                - Very Low Frequency (VLF HRV)
                - Low Frequency (LF HRV)
                - High Frequency (HF HRV)
                - Low Frequency / High Frequency Ratio (LF/HF HRV)
                - Total Power (VLF + LF + HF)
                - Normalised LF (LF HRV %)
                - Normalised HF (HF HRV %)
    """
    fd_features = {}
    timescale = calc_timescale(rr_intervals)
    x = timescale
    y = [rr / 1000 for rr in rr_intervals]

    # Create evenly spaced timeline and resample
    even_x = np.arange(x[0], x[-1], 1 / fs)

    # Cubic interpolation function created
    f = interp1d(x, y, kind='cubic')

    # if PLOT:
    #     y_secs = [rr * 1000 for rr in rr_intervals]
    #     plot_interpolation(x, y_secs, even_x, f)

    # Apply interpolation function
    rr_series = f(even_x)

    # Apply the Welch to estimate the power spectral density (PSD)
    fxx, pxx = signal.welch(x=rr_series, fs=fs, window='hanning', nperseg=256, noverlap=128)

    if PLOT:
        plot_frequency_spectrum(fxx, pxx)

    # Use numpy's Trapezoidal integration function to find area under curve

    # Spacing between points on x-axis
    df = fxx[1] - fxx[0]

    # Splice between 0.0Hz and 0.04Hz for very low frequency range
    fd_features['vlf'] = np.trapz(pxx[(fxx >= 0) & (fxx < 0.04)], dx=df)
    # Splice between 0.04Hz and 0.15Hz for low frequency range
    # LF component that is mediated by both the SNS and PNS
    fd_features['lf'] = np.trapz(pxx[(fxx >= 0.04) & (fxx < 0.15)], dx=df)
    # Splice between 0.15Hz and 0.40Hz for high frequency range
    # HF component that is mediated by the PNS
    fd_features['hf'] = np.trapz(pxx[(fxx >= 0.15) & (fxx < 0.4)], dx=df)

    if normalising:
        denom = ((np.mean(rr_intervals) / 1000) ** 2)
        fd_features['vlf'] = fd_features['vlf'] / denom
        fd_features['lf'] = fd_features['lf'] / denom
        fd_features['hf'] = fd_features['hf'] / denom

    # Calculate total power
    fd_features['total_power'] = fd_features['vlf'] + fd_features['lf'] + fd_features['hf']
    # Calculate the normalised power as a percentage for LF and HF
    fd_features['normlf'] = (fd_features['lf'] / (fd_features['total_power'] - fd_features['vlf'])) * 100
    fd_features['normhf'] = (fd_features['hf'] / (fd_features['total_power'] - fd_features['vlf'])) * 100

    # LF/HF Ratio that is used as an index of autonomic balance
    try:
        fd_features['lf_hf'] = fd_features['lf'] / fd_features['hf']
    except:
        fd_features['lf_hf'] = 0.0
    return fd_features

def remove_ectopic_beats(rr_intervals):
    """ Applies quotient filter to remove ectopic beats, where successive
        intervals will be removed if they differ by more than 20%.

        :returns:
            Corrected RR-intervals with ectopic beats removed

    """
    rr_intervals = np.array(rr_intervals)
    num_intervals = len(rr_intervals) - 1

    indices = np.where(
        (rr_intervals[:num_intervals-1]/rr_intervals[1:num_intervals] < 0.8) | (rr_intervals[:num_intervals-1]/rr_intervals[1:num_intervals] > 1.2) |
        (rr_intervals[1:num_intervals]/rr_intervals[:num_intervals-1] < 0.8) | (rr_intervals[1:num_intervals]/rr_intervals[:num_intervals-1] > 1.2)
    )

    new_intervals = np.delete(rr_intervals, indices)
    return new_intervals
