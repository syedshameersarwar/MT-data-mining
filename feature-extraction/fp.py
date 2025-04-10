from sklearn.mixture import GaussianMixture as GMM

import os
import pandas as pd
import pywt
import numpy as np
import scipy.signal as signal_processing
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class FieldPotentialFeatureExtractor:
    def __init__(
        self,
        case_identifier: str,
        data_dir: str,
        prominence_factor=7.5,
        max_filter_width=0.05,
        max_filter_iter=2,
    ):
        self.case_identifier = case_identifier
        self.data_dir = data_dir
        self.prominence_factor = prominence_factor
        self.max_filter_width = max_filter_width
        self.max_filter_iter = max_filter_iter
        self._validate_paths()

    def _validate_paths(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        if not os.path.exists(
            os.path.join(self.data_dir, f"{self.case_identifier}.hdf")
        ):
            raise FileNotFoundError(
                f"Case {self.case_identifier} does not exist in data directory {self.data_dir}"
            )

    def _apply_dwt_denoising(self, signal):
        DWTcoeffs_full = pywt.wavedec(signal, "db4", mode="reflect")

        DWTcoeffs_full[-1] = np.zeros_like(DWTcoeffs_full[-1])
        DWTcoeffs_full[-2] = np.zeros_like(DWTcoeffs_full[-2])
        DWTcoeffs_full[-3] = pywt.threshold(DWTcoeffs_full[-3], 0.5, mode="soft")
        DWTcoeffs_full[-4] = pywt.threshold(DWTcoeffs_full[-4], 0.5, mode="soft")
        DWTcoeffs_full[-5] = pywt.threshold(DWTcoeffs_full[-5], 0.5, mode="soft")
        DWTcoeffs_full[-6] = pywt.threshold(DWTcoeffs_full[-6], 0.5, mode="soft")
        DWTcoeffs_full[-7] = pywt.threshold(DWTcoeffs_full[-7], 2.5, mode="soft")
        DWTcoeffs_full[-8] = pywt.threshold(DWTcoeffs_full[-8], 2.5, mode="soft")

        filtered = pywt.waverec(DWTcoeffs_full, "db4", mode="reflect")
        filtered_series = pd.Series(filtered[: len(signal)], index=signal.index)
        return filtered_series

    def _apply_max_filter(self, signal, peak_indices):
        sample_rate = 1 / (signal.index[1] - signal.index[0])
        max_filter_width_samples = int(self.max_filter_width * sample_rate)
        source_peaks = peak_indices
        for _ in range(self.max_filter_iter):
            filtered_peaks = []
            for peak in source_peaks:
                # select the absolute maximum value in the range of max_filter_width_samples around the peak
                start, end = np.maximum(0, peak - max_filter_width_samples), np.minimum(
                    len(signal), peak + max_filter_width_samples
                )
                arg_abs_max = signal.iloc[start:end].abs().idxmax()
                int_index = signal.index.get_loc(arg_abs_max)
                filtered_peaks.append(int_index)

            peak_diff_samples = np.diff(filtered_peaks, prepend=0)
            for i in range(1, len(peak_diff_samples)):
                # if difference between two peaks is less than max_filter_width_samples, then remove the peak with lowest amplitude
                if peak_diff_samples[i] < max_filter_width_samples:
                    if (
                        signal.iloc[filtered_peaks[i]]
                        < signal.iloc[filtered_peaks[i - 1]]
                    ):
                        filtered_peaks[i] = -1
                    else:
                        filtered_peaks[i - 1] = -1
            # drop duplicates and sort peak indices
            filtered_peaks = [peak for peak in filtered_peaks if peak != -1]
            source_peaks = sorted(list(set(filtered_peaks)))
        return source_peaks

    def _find_fp_peaks_using_gmm_fit(self, signal):
        X = signal.values.reshape(-1, 1)
        gmm = GMM(
            n_components=4, max_iter=1000, random_state=42, covariance_type="full"
        )
        fit = gmm.fit(X)
        mean = fit.means_
        covs = fit.covariances_

        means = [mean[0][0], mean[1][0], mean[2][0], mean[3][0]]
        stds = [
            np.sqrt(covs[0][0][0]),
            np.sqrt(covs[1][0][0]),
            np.sqrt(covs[2][0][0]),
            np.sqrt(covs[3][0][0]),
        ]

        highest_std = np.argmax(stds)
        # remove the gaussian with the highest std
        means.pop(highest_std)
        stds.pop(highest_std)

        lowest_std = np.argmin(stds)
        prominence = np.round(means[lowest_std], 4) + self.prominence_factor * np.round(
            stds[lowest_std], 4
        )
        peaks, _ = signal_processing.find_peaks(abs(signal), prominence=prominence)

        filtered_peaks = self._apply_max_filter(signal, peaks)
        return filtered_peaks

    def _determine_peak_type(self, signal, peak_indices):
        peak_width_result = signal_processing.peak_widths(
            np.abs(signal), peak_indices, rel_height=0.5
        )
        peak_widths = peak_width_result[0] * (signal.index[1] - signal.index[0])

        peaks_df = pd.DataFrame()
        peaks_df["amplitude"] = signal.iloc[peak_indices]
        peaks_df["peaks_iloc"] = peak_indices
        peaks_df["peak_time_idx"] = signal.iloc[peak_indices].index.values
        peaks_df["type"] = "T"

        # for all peaks having half width half maximum less than 0.01, set type to Na+
        peaks_df.loc[peak_widths < 0.01, "type"] = "Na+"
        # for all peaks having amplitude greater than 0, set type to T+
        peaks_df.loc[(peaks_df.type == "T") & (peaks_df.amplitude > 0), "type"] = "T+"
        # for all peaks having amplitude less than 0, set type to T-
        peaks_df.loc[(peaks_df.type == "T") & (peaks_df.amplitude < 0), "type"] = "T-"
        # for all peaks which are assigned T- and have half width half maximum <= 0.01 as Na+ (covering sodium spikes without positive part)
        peaks_df.loc[
            (peaks_df.type == "T-") & (np.round(peak_widths, 2) <= 0.01), "type"
        ] = "Na+"
        return peaks_df, peak_width_result

    def _plot_field_potential_with_peaks(
        self, signal, peak_types, peak_width_result, frequency, show=False
    ):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        peaks = peak_types.peaks_iloc.values

        fig.add_trace(
            go.Scatter(
                x=signal.index,
                y=signal,
                mode="lines",
                name="Raw Signal",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=signal.index,
                y=signal,
                mode="lines",
                name="Filtered Signal",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=signal.index,
                y=signal,
                mode="lines",
                name="Filtered Signal",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=signal.iloc[peaks].index,
                y=signal.iloc[peaks],
                mode="markers",
                name="New Peaks",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=signal.loc[peak_types[peak_types.type == "Na+"].peak_time_idx].index,
                y=signal.loc[peak_types[peak_types.type == "Na+"].peak_time_idx],
                mode="markers",
                name="Na+ Peaks",
                marker=dict(color="red", size=5),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=signal.loc[peak_types[peak_types.type == "T+"].peak_time_idx].index,
                y=signal.loc[peak_types[peak_types.type == "T+"].peak_time_idx],
                mode="markers",
                name="T+ Peaks",
                marker=dict(color="orange", size=5),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=signal.loc[peak_types[peak_types.type == "T-"].peak_time_idx].index,
                y=signal.loc[peak_types[peak_types.type == "T-"].peak_time_idx],
                mode="markers",
                name="T- Peaks",
                marker=dict(color="black", size=5),
            ),
            row=2,
            col=1,
        )

        for i in range(len(peak_width_result[1:][0])):
            sign = np.sign(signal.iloc[peaks[i]])
            x0 = signal.iloc[[int(np.round(peak_width_result[2][i], 0))]].index[0]
            y0 = sign * np.round(peak_width_result[1][i], 4)
            x1 = signal.iloc[[int(np.round(peak_width_result[3][i], 0))]].index[0]
            y1 = sign * np.round(peak_width_result[1][i], 4)

            fig.add_shape(
                dict(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color="red", width=1),
                ),
                row=2,
                col=1,
            )
        fig.update_layout(
            title_text=f"Case: {self.case_identifier} - Global Frequency: {frequency} Hz",
            autosize=False,
            width=2000,
            height=1500,
        )

        if show:
            fig.show()

        return fig

    def _extract_fpd_feature(self, signal, peak_types):
        fpd_df = pd.DataFrame()
        na_peaks = peak_types[peak_types.type == "Na+"]
        t_pos_peaks = peak_types[peak_types.type == "T+"]

        # select all the Na peaks and take their time as the start of the contraction-event
        fpd_df["event_index"] = range(na_peaks.shape[0])
        fpd_df["start_time"] = na_peaks.peak_time_idx.values

        if t_pos_peaks.shape[0] == 0:
            print(
                f"No T+ waves found for {self.case_identifier}, Cannot proceed with feature extraction..."
            )
            return

        drop_indices = []
        for i in range(len(fpd_df)):
            event_start = fpd_df.loc[i, "start_time"]
            next_na_peak_time_idx = (
                fpd_df.loc[i + 1, "start_time"]
                if i + 1 < len(fpd_df)
                else signal.index[-1]
            )

            # find all the T+ peaks found in time range of [current Na+ peak, next Na+ peak - 0.15 seconds]
            t_pos_peaks_in_interval = t_pos_peaks[
                (t_pos_peaks.peak_time_idx > event_start)
                & (
                    np.round(t_pos_peaks.peak_time_idx, 4)
                    < np.round(next_na_peak_time_idx - 0.15, 4)
                )
            ]

            if t_pos_peaks_in_interval.shape[0] == 0:
                # drop the peaks without any T+ wave detected
                drop_indices.append(i)
                continue

            # select the  time of last T+ wave detected in above interval as the end of the contraction-event
            event_end = t_pos_peaks_in_interval["peak_time_idx"].values[-1]
            fpd_df.loc[i, "end_time"] = event_end
            fpd_df.loc[i, "na_peak_index"] = int(na_peaks["peaks_iloc"].values[i])
            fpd_df.loc[i, "t_peak_index"] = int(
                t_pos_peaks_in_interval["peaks_iloc"].values[-1]
            )
            fpd_df.loc[i, "duration"] = np.round(event_end - event_start, 3)

        # if last event does not have T+ wave, discard it in frequency calculation
        if np.isnan(fpd_df.iloc[-1]["end_time"]):
            fpd_df = fpd_df.drop(drop_indices)
            freqeuncy = np.round(len(fpd_df) / fpd_df.iloc[-1]["end_time"], 2)
        else:
            freqeuncy = np.round(len(fpd_df) / fpd_df.iloc[-1]["end_time"], 2)
            # drop events without detected T+ wave, but included them in frequency calculation
            fpd_df = fpd_df.drop(drop_indices)

        fpd_df["na_peak_index"] = pd.to_numeric(
            fpd_df["na_peak_index"], downcast="integer"
        )
        fpd_df["t_peak_index"] = pd.to_numeric(
            fpd_df["t_peak_index"], downcast="integer"
        )
        return fpd_df, freqeuncy

    def extract(self):
        df = pd.read_hdf(os.path.join(self.data_dir, f"{self.case_identifier}.hdf"))
        signal = df.mea
        denoised_signal = self._apply_dwt_denoising(signal)
        fp_peaks = self._find_fp_peaks_using_gmm_fit(denoised_signal)
        peak_types_df, peak_width_result = self._determine_peak_type(
            denoised_signal, fp_peaks
        )
        fpd_feature_result = self._extract_fpd_feature(denoised_signal, peak_types_df)
        if fpd_feature_result is None:
            return
        fpd_df, frequency = fpd_feature_result
        fig = self._plot_field_potential_with_peaks(
            signal, peak_types_df, peak_width_result, frequency
        )
        return fpd_df, denoised_signal, frequency, fig
