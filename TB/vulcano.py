import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f_oneway

from adjustText import adjust_text



class Vulcano:
    def __init__(self, test_func=None):
        if test_func is None:
            self.test_func = f_oneway
        else:
            self.test_func = test_func
        self.results = None
        self.group_labels = None
        self.significance_base_level = 0.05
        self.significance_threshold = None

    def fit(self, X, y):

        data = pd.DataFrame(X)
        labels = list(y)

        features = data.columns.to_list()

        self.group_labels = self.get_group_labels(labels)
        label_0, label_1 = self.group_labels

        data["labels"] = labels

        grps = data.groupby("labels")

        grp_0, grp_1 = grps.get_group(label_0), grps.get_group(label_1)

        p_values = []
        fold_changes = []

        results = pd.DataFrame()

        for feature in features:
            try:
                p_value = self.calculate_p_value(grp_0[feature], grp_1[feature])
                fold_change = self.calculate_fold_change(grp_0[feature], grp_1[feature])
                results.loc[feature, ["p-value", "fold-change"]] = p_value, fold_change
            except ZeroDivisionError as e:
                results.loc[feature, ["p-value", "fold-change"]] = 1, 1

        results["-log10(p-value)"] = -results["p-value"].apply(np.log10)
        results["log2(fold-change)"] = results["fold-change"].apply(np.log2)
        results.index.name = "Feature"

        n_features = len(features)

        self.significance_threshold = self.calculate_significance_threshold(
            self.significance_base_level, n_features
        )

        results["significance_threshold"] = self.significance_threshold
        results["Significant"] = results["p-value"] < self.significance_threshold

        self.results = results.reset_index()
        return self.results

    def get_group_labels(self, labels):
        group_labels = list(set(labels))
        group_labels.sort()
        assert len(group_labels) == 2
        return (group_labels[0], group_labels[1])

    def calculate_significance_threshold(self, base_level, n_tials):
        return self.significance_base_level / n_tials

    def calculate_p_value(self, values_0, values_1):
        return self.test_func(values_0, values_1).pvalue

    def calculate_fold_change(self, values_0, values_1):
        return np.mean(values_1) / np.mean(values_0)

    def plot_interactive(self, height=750, width=750):
        results = self.results
        fig = px.scatter(
            data_frame=results,
            y="-log10(p-value)",
            x="log2(fold-change)",
            hover_data=["Feature", "p-value", "fold-change"],
            height=height,
            width=width,
            color="Significant",
        )
        fig.update_traces(
            marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )
        fig.add_hline(
            y=-np.log10(self.significance_threshold), line_width=0.5, line_dash="dash"
        )
        fig.add_vline(x=0, line_width=0.5, line_dash="dash")

        sig = self.results[self.results.Significant]

        fig.add_trace(
            go.Scatter(
                x=sig["log2(fold-change)"],
                y=sig["-log10(p-value)"],
                mode="text",
                text=sig.Feature,
                name="Feature",
                textposition="top center",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.array(
                    [
                        results["log2(fold-change)"].min(),
                        results["log2(fold-change)"].max(),
                    ]
                ),
                y=np.array(
                    [results["-log10(p-value)"].max(), results["-log10(p-value)"].max()]
                )
                * 1.1,
                mode="text",
                text=self.group_labels,
                name="Group",
                textposition="middle center",
                textfont=dict(color="crimson"),
                marker=dict(size=3),
                hoverinfo="skip",
            )
        )

        return fig

    def plot(self, minfoldchange=1, nmaxannot=None, legend=False, **kwargs):

        x = "log2(fold-change)"
        y = "-log10(p-value)"

        results = self.results.copy()
        n_results = len(results)
        results["_colors_"] = get_colors(
            results.Significant, results[x], [minfoldchange] * n_results
        )

        g = sns.scatterplot(
            data=results, x=x, y=y, hue="_colors_", legend=legend, **kwargs
        )

        plt.axhline(
            y=-np.log10(self.significance_threshold), lw=0.5, ls="--", color="k"
        )

        plt.axvline(x=minfoldchange, lw=0.5, ls="--", color="k")
        plt.axvline(x=-minfoldchange, lw=0.5, ls="--", color="k")

        x_groups = [0.015, 0.985]
        y_groups = [0.03, 0.03]
        halign = ["left", "right"]

        ax = plt.gca()
        # Add group labels

        for i in [0, 1]:
            text = plt.text(
                x_groups[i],
                y_groups[i],
                self.group_labels[i],
                color=".3",
                horizontalalignment=halign[i],
                transform=ax.transAxes,
                backgroundcolor="0.3",
                bbox=dict(facecolor="none", edgecolor="0.3", boxstyle="round, pad=0.2"),
            )

        sig = results[results.Significant].sort_values("p-value")

        if nmaxannot is not None:
            sig = sig.head(nmaxannot)

        # Add labels
        texts = []
        for ndx, row in sig.iterrows():
            x_value = row[x]
            y_value = row[y]
            _text = row["Feature"]
            if np.abs(row["log2(fold-change)"]) < minfoldchange:
                continue
            text = plt.text(
                x_value, y_value, _text, color="black", horizontalalignment="center"
            )
            texts.append(text)

        adjust_text(texts, arrowprops=dict(arrowstyle="->", color="k", lw=0.5))

        sns.despine()


def _get_color_(sig, log_fc, minfoldchange=1):
    if sig:
        if log_fc >= minfoldchange:
            return "C2"
        elif log_fc <= -minfoldchange:
            return "C3"
    return "C1"


get_colors = np.vectorize(_get_color_)

