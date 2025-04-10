import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scipy.signal as signal_processing
from utils.series import index_to_xdata
from scipy import stats
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils.preprocessing as preprocessing
import json


class CommonFeatureExtractor:
    def __init__(
        self,
        case_identifier: str,
        signal_type: str = "force",
        data_dir: str = "/home/syedshameersarwar/Desktop/Work/MyOfarm/Internship/Feature Extracion & Clustering/arrythmia-prod/Preprocessed/HDFs",
        peaks_dir: str = "/home/syedshameersarwar/Desktop/Work/MyOfarm/Internship/Feature Extracion & Clustering/arrythmia-prod/Preprocessed/Peaks",
    ):
        self.case_identifier = case_identifier
        self.signal_type = signal_type
        self.data_dir = data_dir
        self.peaks_dir = peaks_dir
        if signal_type not in ["force", "calcium"]:
            raise ValueError("Invalid signal type, must be either 'force' or 'calcium'")
        self._validate_paths()

    def _validate_paths(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        if not os.path.exists(self.peaks_dir):
            raise FileNotFoundError(f"Peaks directory {self.peaks_dir} does not exist")
        if not os.path.exists(
            os.path.join(self.data_dir, f"{self.case_identifier}.hdf")
        ):
            raise FileNotFoundError(
                f"Case {self.case_identifier} does not exist in data directory {self.data_dir}"
            )
        if not os.path.exists(
            os.path.join(self.peaks_dir, f"{self.case_identifier}.json")
        ):
            raise FileNotFoundError(
                f"Case {self.case_identifier} does not exist in peaks directory {self.peaks_dir}"
            )

    def _get_width_diff_features_at_diffs(
        self,
        signal,
        peak_indices,
        relative_height=0.5,
        width_only_features=False,
        is_first=False,
    ):
        rel_height_low = signal_processing.peak_widths(
            signal, peak_indices, rel_height=relative_height
        )
        rel_height_high = signal_processing.peak_widths(
            signal, peak_indices, rel_height=np.round(1 - relative_height, 2)
        )

        left_ips_high = index_to_xdata(signal.index, rel_height_high[2])
        right_ips_high = index_to_xdata(signal.index, rel_height_high[3])
        left_ips_low = index_to_xdata(signal.index, rel_height_low[2])
        right_ips_low = index_to_xdata(signal.index, rel_height_low[3])

        widths = np.round(
            rel_height_high[0] * np.round(signal.index[1] - signal.index[0], 4), 2
        )

        rise_time = np.round(np.abs(left_ips_high - left_ips_low), 2)
        decay_time = np.round(np.abs(right_ips_high - right_ips_low), 2)
        if not width_only_features:
            rise_time_points = [
                (
                    signal[
                        np.round(left_ips_low[i], 4) : np.round(left_ips_high[i], 4)
                    ].index[0],
                    signal[
                        np.round(left_ips_low[i], 4) : np.round(left_ips_high[i], 4)
                    ].index[-1],
                )
                for i in range(len(peak_indices))
            ]
            decay_time_points = [
                (
                    signal[
                        np.round(right_ips_high[i], 4) : np.round(right_ips_low[i], 4)
                    ].index[0],
                    signal[
                        np.round(right_ips_high[i], 4) : np.round(right_ips_low[i], 4)
                    ].index[-1],
                )
                for i in range(len(peak_indices))
            ]
        width_features_df = pd.DataFrame()
        width_features_df["peaks_iloc"] = peak_indices

        if is_first:
            width_features_df["peaks_time_idx"] = signal.iloc[peak_indices].index.values
            width_features_df["peak_amplitude"] = np.round(
                signal.iloc[peak_indices].values, 3
            )

        width_features_df[f"width_{relative_height} s"] = widths
        if not width_only_features:
            width_features_df[
                f"rise_time_{np.round(1 - relative_height, 2)}_{relative_height} s"
            ] = rise_time
            width_features_df[
                f"decay_time_{np.round(1 - relative_height, 2)}_{relative_height} s"
            ] = decay_time
            width_features_df[
                f"rise_time_points_{np.round(1 - relative_height, 2)}_{relative_height} s"
            ] = rise_time_points
            width_features_df[
                f"decay_time_points_{np.round(1 - relative_height, 2)}_{relative_height} s"
            ] = decay_time_points

        width_features_df[f"ips_left_high_{relative_height}"] = rel_height_high[2]
        width_features_df[f"ips_right_high_{relative_height}"] = rel_height_high[3]
        width_features_df[f"ips_left_low_{relative_height}"] = rel_height_low[2]
        width_features_df[f"ips_right_low_{relative_height}"] = rel_height_low[3]
        width_features_df[f"ips_height_high_{relative_height}"] = rel_height_high[1]
        width_features_df[f"ips_height_low_{relative_height}"] = rel_height_low[1]

        return width_features_df

    def _get_width_abs_diff_features_at_relative_height(
        self, signal, peak_indices, relative_height=0.2
    ):
        rel_height_low = signal_processing.peak_widths(
            signal, peak_indices, rel_height=np.round(1 - relative_height, 2)
        )
        left_ips_low = index_to_xdata(signal.index, rel_height_low[2])
        right_ips_low = index_to_xdata(signal.index, rel_height_low[3])

        rise_time_abs = np.round(
            np.abs(left_ips_low - signal.iloc[peak_indices].index.values), 2
        )
        rise_time_abs_points = [
            (
                signal[
                    np.round(left_ips_low[i], 4) : signal.iloc[peak_indices].index[i]
                ].index[0],
                signal[
                    np.round(left_ips_low[i], 4) : signal.iloc[peak_indices].index[i]
                ].index[-1],
            )
            for i in range(len(peak_indices))
        ]

        decay_time_abs = np.round(
            np.abs(signal.iloc[peak_indices].index.values - right_ips_low), 2
        )
        decay_time_abs_points = [
            (
                signal[
                    signal.iloc[peak_indices].index[i] : np.round(right_ips_low[i], 4)
                ].index[0],
                signal[
                    signal.iloc[peak_indices].index[i] : np.round(right_ips_low[i], 4)
                ].index[-1],
            )
            for i in range(len(peak_indices))
        ]

        abs_diff_features_df = pd.DataFrame()
        abs_diff_features_df["peaks_iloc"] = peak_indices
        abs_diff_features_df[f"rise_time_{relative_height}_max s"] = rise_time_abs
        abs_diff_features_df[f"decay_time_max_{relative_height} s"] = decay_time_abs
        abs_diff_features_df[f"rise_time_points_{relative_height}_max s"] = (
            rise_time_abs_points
        )
        abs_diff_features_df[f"decay_time_points_max_{relative_height} s"] = (
            decay_time_abs_points
        )
        abs_diff_features_df[f"ips_abs_left_low_{relative_height}"] = rel_height_low[2]
        abs_diff_features_df[f"ips_abs_right_low_{relative_height}"] = rel_height_low[3]
        abs_diff_features_df[f"ips_abs_height_low_{relative_height}"] = rel_height_low[
            1
        ]

        return abs_diff_features_df

    def _plot_features(
        self,
        signal,
        feature_df,
        width_only_features=[0.5, 0.2],
        diffs=[0.8],
        max_diffs=[0.8],
        show=False,
    ):
        titles = (
            f"{self.signal_type.title()} Signal with Peaks",
            f"Width {width_only_features[0]} [{np.round(feature_df[f'width_{width_only_features[0]} s'].mean(),2)} \u00B1 {np.round(feature_df[f'width_{width_only_features[0]} s'].std(), 2)}] s",
            f"Width {width_only_features[1]} [{np.round(feature_df[f'width_{width_only_features[1]} s'].mean(),2)} \u00B1 {np.round(feature_df[f'width_{width_only_features[1]} s'].std(), 2)}] s",
        )
        for i in range(len(diffs)):
            titles += (
                f"Width {diffs[i]} [Width={np.round(feature_df[f'width_{diffs[i]} s'].mean(),2)} \u00B1 {np.round(feature_df[f'width_{diffs[i]} s'].std(),2)}, RT={np.round(feature_df[f'rise_time_{np.round(1-diffs[i],2)}_{diffs[i]} s'].mean(),2)} \u00B1 {np.round(feature_df[f'rise_time_{np.round(1-diffs[i],2)}_{diffs[i]} s'].std(), 2)} s, DT={np.round(feature_df[f'decay_time_{np.round(1-diffs[i],2)}_{diffs[i]} s'].mean(),2)} \u00B1 {np.round(feature_df[f'decay_time_{np.round(1-diffs[i],2)}_{diffs[i]} s'].std(), 2)} s]",
            )
        for i in range(len(max_diffs)):
            titles += (
                f"Abs diff {max_diffs[i]} [RT={np.round(feature_df[f'rise_time_{max_diffs[i]}_max s'].mean(),2)} \u00B1 {np.round(feature_df[f'rise_time_{max_diffs[i]}_max s'].std(), 2)} s, DT={np.round(feature_df[f'decay_time_max_{max_diffs[i]} s'].mean(),2)} \u00B1 {np.round(feature_df[f'decay_time_max_{max_diffs[i]} s'].std(), 2)} s]",
            )
        fig = make_subplots(
            rows=3 + len(diffs) + len(max_diffs),
            cols=1,
            shared_xaxes=True,
            subplot_titles=titles,
        )

        # Plot 1 - signal with peaks, also indicating the outliers
        fig.add_trace(
            go.Scatter(
                x=signal.index,
                y=signal.values,
                mode="lines",
                name=self.signal_type.title(),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=signal.index[feature_df.peaks_iloc],
                y=signal.values[feature_df.peaks_iloc],
                mode="markers",
                marker=dict(color="red"),
                name="Peaks",
            ),
            row=1,
            col=1,
        )
        # plot outliers
        fig.add_trace(
            go.Scatter(
                x=signal.index[
                    feature_df[feature_df.potential_outlier == 1].peaks_iloc
                ],
                y=signal.values[
                    feature_df[feature_df.potential_outlier == 1].peaks_iloc
                ],
                mode="markers",
                name="Outliers",
                marker=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        # Plot 2 - width 0.5 - width_only[0]
        fig.add_trace(
            go.Scatter(x=signal.index, y=signal.values, mode="lines"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=signal.index[feature_df.peaks_iloc],
                y=signal.values[feature_df.peaks_iloc],
                mode="markers",
                marker=dict(color="red"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=signal.index[
                    feature_df[feature_df.potential_outlier == 1].peaks_iloc
                ],
                y=signal.values[
                    feature_df[feature_df.potential_outlier == 1].peaks_iloc
                ],
                mode="markers",
                marker=dict(color="blue"),
            ),
            row=2,
            col=1,
        )
        for i in range(len(feature_df.peaks_iloc)):
            sign = np.sign(signal.iloc[feature_df.peaks_iloc.values[i]])
            x0 = signal.iloc[
                [int(feature_df[f"ips_left_high_{width_only_features[0]}"].values[i])]
            ].index[0]
            y0 = sign * np.round(
                feature_df[f"ips_height_high_{width_only_features[0]}"].values[i], 4
            )
            x1 = signal.iloc[
                [int(feature_df[f"ips_right_high_{width_only_features[0]}"].values[i])]
            ].index[0]
            y1 = sign * np.round(
                feature_df[f"ips_height_high_{width_only_features[0]}"].values[i], 4
            )

            fig.add_shape(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color="green", width=1),
                row=2,
                col=1,
            )

        # Plot 3 - width 0.2 - width_only[1]
        fig.add_trace(
            go.Scatter(x=signal.index, y=signal.values, mode="lines"),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=signal.index[feature_df.peaks_iloc],
                y=signal.values[feature_df.peaks_iloc],
                mode="markers",
                marker=dict(color="red"),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=signal.index[
                    feature_df[feature_df.potential_outlier == 1].peaks_iloc
                ],
                y=signal.values[
                    feature_df[feature_df.potential_outlier == 1].peaks_iloc
                ],
                mode="markers",
                marker=dict(color="blue"),
            ),
            row=3,
            col=1,
        )

        for i in range(len(feature_df.peaks_iloc)):
            sign = np.sign(signal.iloc[feature_df.peaks_iloc.values[i]])
            x0 = signal.iloc[
                [int(feature_df[f"ips_left_high_{width_only_features[1]}"].values[i])]
            ].index[0]
            y0 = sign * np.round(
                feature_df[f"ips_height_high_{width_only_features[1]}"].values[i], 4
            )
            x1 = signal.iloc[
                [int(feature_df[f"ips_right_high_{width_only_features[1]}"].values[i])]
            ].index[0]
            y1 = sign * np.round(
                feature_df[f"ips_height_high_{width_only_features[1]}"].values[i], 4
            )

            fig.add_shape(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color="green", width=1),
                row=3,
                col=1,
            )

        # Plot 4 - width 0.8 - relative_times with rise time and decay time
        for i in range(len(diffs)):
            relative_height = diffs[i]
            row = 4 + i
            fig.add_trace(
                go.Scatter(x=signal.index, y=signal.values, mode="lines"),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=signal.index[feature_df.peaks_iloc],
                    y=signal.values[feature_df.peaks_iloc],
                    mode="markers",
                    marker=dict(color="red"),
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=signal.index[
                        feature_df[feature_df.potential_outlier == 1].peaks_iloc
                    ],
                    y=signal.values[
                        feature_df[feature_df.potential_outlier == 1].peaks_iloc
                    ],
                    mode="markers",
                    marker=dict(color="blue"),
                ),
                row=row,
                col=1,
            )

            # line 1 - width 0.8
            for i in range(len(feature_df.peaks_iloc)):
                sign = np.sign(signal.iloc[feature_df.peaks_iloc.values[i]])
                x0 = signal.iloc[
                    [int(feature_df[f"ips_left_high_{relative_height}"].values[i])]
                ].index[0]
                y0 = sign * np.round(
                    feature_df[f"ips_height_high_{relative_height}"].values[i], 4
                )
                x1 = signal.iloc[
                    [int(feature_df[f"ips_right_high_{relative_height}"].values[i])]
                ].index[0]
                y1 = sign * np.round(
                    feature_df[f"ips_height_high_{relative_height}"].values[i], 4
                )

                fig.add_shape(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color="green", width=1),
                    row=row,
                    col=1,
                )
            # line 2 - width 0.2 := ips_left_low, ips_right_low
            for i in range(len(feature_df.peaks_iloc)):
                sign = np.sign(signal.iloc[feature_df.peaks_iloc.values[i]])
                x0 = signal.iloc[
                    [int(feature_df[f"ips_left_low_{relative_height}"].values[i])]
                ].index[0]
                y0 = sign * np.round(
                    feature_df[f"ips_height_low_{relative_height}"].values[i], 4
                )
                x1 = signal.iloc[
                    [int(feature_df[f"ips_right_low_{relative_height}"].values[i])]
                ].index[0]
                y1 = sign * np.round(
                    feature_df[f"ips_height_low_{relative_height}"].values[i], 4
                )

                fig.add_shape(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color="green", width=1),
                    row=row,
                    col=1,
                )

            # line 3 - 1st vertical line for rise time - from ips_left_low to min of signal
            for i in range(len(feature_df.peaks_iloc)):
                sign = np.sign(signal.iloc[feature_df.peaks_iloc.values[i]])
                x0 = signal.iloc[
                    [int(feature_df[f"ips_left_low_{relative_height}"].values[i])]
                ].index[0]
                y0 = sign * np.round(
                    feature_df[f"ips_height_low_{relative_height}"].values[i], 4
                )
                x1 = x0
                y1 = signal.min()

                fig.add_shape(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x0,
                    y1=y1,
                    line=dict(color="black", width=0.5),
                    row=row,
                    col=1,
                )

            # line 4 - 2nd vertical line for rise time - from ips_left_high to min of signal
            for i in range(len(feature_df.peaks_iloc)):
                sign = np.sign(signal.iloc[feature_df.peaks_iloc.values[i]])
                x0 = signal.iloc[
                    [int(feature_df[f"ips_left_high_{relative_height}"].values[i])]
                ].index[0]
                y0 = sign * np.round(
                    feature_df[f"ips_height_high_{relative_height}"].values[i], 4
                )
                x1 = x0
                y1 = signal.min()

                fig.add_shape(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x0,
                    y1=y1,
                    line=dict(color="black", width=0.5),
                    row=row,
                    col=1,
                )

            # line 5 - 1st vertical line for decay time - from ips_right_high to min of signal
            for i in range(len(feature_df.peaks_iloc)):
                sign = np.sign(signal.iloc[feature_df.peaks_iloc.values[i]])
                x0 = signal.iloc[
                    [int(feature_df[f"ips_right_high_{relative_height}"].values[i])]
                ].index[0]
                y0 = sign * np.round(
                    feature_df[f"ips_height_high_{relative_height}"].values[i], 4
                )
                x1 = x0
                y1 = signal.min()

                fig.add_shape(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x0,
                    y1=y1,
                    line=dict(color="purple", width=0.5),
                    row=row,
                    col=1,
                )

            # line 6 - 2nd vertical line for decay time - from ips_right_low to min of signal
            for i in range(len(feature_df.peaks_iloc)):
                sign = np.sign(signal.iloc[feature_df.peaks_iloc.values[i]])
                x0 = signal.iloc[
                    [int(feature_df[f"ips_right_low_{relative_height}"].values[i])]
                ].index[0]
                y0 = sign * np.round(
                    feature_df[f"ips_height_low_{relative_height}"].values[i], 4
                )
                x1 = x0
                y1 = signal.min()

                fig.add_shape(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x0,
                    y1=y1,
                    line=dict(color="purple", width=0.5),
                    row=row,
                    col=1,
                )

        # Plot 5 - abs diff 0.8 - abs_height
        for i in range(len(max_diffs)):
            abs_height = max_diffs[i]
            row = 4 + len(diffs) + i
            fig.add_trace(
                go.Scatter(x=signal.index, y=signal.values, mode="lines"),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=signal.index[feature_df.peaks_iloc],
                    y=signal.values[feature_df.peaks_iloc],
                    mode="markers",
                    marker=dict(color="red"),
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=signal.index[
                        feature_df[feature_df.potential_outlier == 1].peaks_iloc
                    ],
                    y=signal.values[
                        feature_df[feature_df.potential_outlier == 1].peaks_iloc
                    ],
                    mode="markers",
                    marker=dict(color="blue"),
                ),
                row=row,
                col=1,
            )

            # line 1 - vertical line from ips_left_low to signal min for rise time
            for i in range(len(feature_df.peaks_iloc)):
                sign = np.sign(signal.iloc[feature_df.peaks_iloc.values[i]])
                x0 = signal.iloc[
                    [int(feature_df[f"ips_abs_left_low_{abs_height}"].values[i])]
                ].index[0]
                y0 = sign * np.round(
                    feature_df[f"ips_abs_height_low_{abs_height}"].values[i], 4
                )
                x1 = x0
                y1 = signal.min()

                fig.add_shape(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color="green", width=0.5),
                    row=row,
                    col=1,
                )
            # line 2 - vertical line from signal peak to signal min
            for i in range(len(feature_df.peaks_iloc)):
                sign = np.sign(signal.iloc[feature_df.peaks_iloc.values[i]])
                x0 = signal.iloc[[feature_df.peaks_iloc.values[i]]].index[0]
                y0 = sign * signal.iloc[feature_df.peaks_iloc.values[i]]
                x1 = x0
                y1 = signal.min()

                fig.add_shape(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color="black", width=0.5),
                    row=row,
                    col=1,
                )

            # line 3 - vertical line from ips_right_low to signal min for decay time
            for i in range(len(feature_df.peaks_iloc)):
                sign = np.sign(signal.iloc[feature_df.peaks_iloc.values[i]])
                x0 = signal.iloc[
                    [int(feature_df[f"ips_abs_right_low_{abs_height}"].values[i])]
                ].index[0]
                y0 = sign * np.round(
                    feature_df[f"ips_abs_height_low_{abs_height}"].values[i], 4
                )
                x1 = x0
                y1 = signal.min()

                fig.add_shape(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color="purple", width=0.5),
                    row=row,
                    col=1,
                )
        fig.update_layout(
            title=f"{self.signal_type.title()}:  {self.case_identifier } - Width Features",
            autosize=False,
            width=1200,
            height=1000,
        )
        if show:
            fig.show()
        return fig

    def extract(self):
        hdf_file_path = os.path.join(self.data_dir, f"{self.case_identifier}.hdf")
        peaks_file_path = os.path.join(self.peaks_dir, f"{self.case_identifier}.json")

        df = pd.read_hdf(hdf_file_path)
        if self.signal_type == "force":
            signal = df.force
            peaks = json.load(open(peaks_file_path))["force_peaks_indexes"]
        else:
            signal = df.calc
            peaks = json.load(open(peaks_file_path))["calc_peaks_indexes"]
        feature_dict = {}
        # width at relative heights of 0.5 and 0.2
        for i, height in enumerate([0.5, 0.2]):
            feature_dict[f"width_features_{height}"] = (
                self._get_width_diff_features_at_diffs(
                    signal,
                    peaks,
                    height,
                    width_only_features=True,
                    is_first=True if i == 0 else False,
                )
            )

        # rise_time/decay_time between [height,1-height] and width and 0.8
        feature_dict["rt_dt_width_0.8"] = self._get_width_diff_features_at_diffs(
            signal, peaks, relative_height=0.8
        )

        # rise_time/decay_time between [height,max-height]
        feature_dict["rt_dt_max_0.8"] = (
            self._get_width_abs_diff_features_at_relative_height(
                signal, peaks, relative_height=0.8
            )
        )

        # find outlier peaks on the basis of width at Relative  height of 0.2
        z = np.abs(stats.zscore(feature_dict["width_features_0.2"]["width_0.2 s"]))
        # Identify outliers as students with a z-score greater than 3
        threshold = 3
        outliers = feature_dict["width_features_0.2"][z > threshold]

        feature_keys = list(feature_dict.keys())
        feature_df = feature_dict[feature_keys[0]]
        for i in range(1, len(feature_keys)):
            feature_df = pd.merge(
                feature_df,
                feature_dict[feature_keys[i]],
                on="peaks_iloc",
            )
        feature_df["potential_outlier"] = 0
        feature_df.loc[outliers.index, "potential_outlier"] = 1
        fig = self._plot_features(
            signal,
            feature_df,
            width_only_features=[0.5, 0.2],
            diffs=[0.8],
            max_diffs=[0.8],
        )
        # drop all columns that start with ips_
        feature_df = feature_df.drop(
            columns=[col for col in feature_df.columns if col.startswith("ips_")]
        )

        return feature_df, fig
