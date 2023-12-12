from typing import List, Optional, Tuple, Union

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def adjust_lightness(color, amount=1.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:  # noqa: E722
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def plot_hist(data: List[NDArray],
              fig_data: dict,
              bins: Optional[Union[List[List[float]], List[int]]] = None,
              norm_pdf: bool = False,
              count: bool = False) -> Tuple[plt.Figure, plt.Axes]:

    # print histograms with normal distribution if required
    if "figsize_factor" in fig_data:
        wf, hf = fig_data["figsize_factor"]
    else:
        wf, hf = (1.2, 1.3)

    fig_w, fig_h = plt.rcParamsDefault["figure.figsize"]
    figsize = (fig_w * wf * len(data), fig_h * hf)
    figs, axes = plt.subplots(nrows=1,
                              ncols=len(data),
                              figsize=figsize,
                              squeeze=False)
    axes_ = np.array(axes).reshape(-1)

    # get color cycles
    hist_colors = plt.get_cmap("Accent")
    line_colors = plt.get_cmap("tab10")
    text_colors = plt.get_cmap("Set1")
    # plot histogram for each data
    for i, (ax, d) in enumerate(zip(axes_, data)):
        if bins is None:
            bins = [30] * len(data)
        density, _bins, _ = ax.hist(d,
                                    bins=bins[i],
                                    density=True,
                                    alpha=0.5,
                                    color=hist_colors(i),
                                    ec=adjust_lightness(hist_colors(i)),
                                    label=fig_data["label_h"][i])

        _ = ax.set_xticks(_bins)
        _ = ax.set_xticklabels([str(round(float(b), 5)) for b in _bins],
                               rotation=90)

        # show counts on hist
        if count:
            counts, _ = np.histogram(d, _bins)
            Xs = [(e + s) / 2 for s, e in zip(_bins[:-1], _bins[1:])]
            for x, y, count in zip(Xs, density, counts):
                _ = ax.text(x,
                            y * 1.02,
                            count,
                            horizontalalignment="center",
                            rotation=45,
                            color=text_colors(i))

        # plot normal probability dist
        if norm_pdf:
            # calc normal distribution of bleu4
            d_sorted = np.sort(d)
            mu = np.mean(d)
            sig = np.std(d)
            data_norm_pdf = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(
                -np.power((d_sorted - mu) / sig, 2.) / 2)

            _ = ax.plot(d_sorted,
                        data_norm_pdf,
                        color=line_colors(i),
                        linestyle="--",
                        linewidth=2,
                        label=fig_data["label_l"][i])

        _ = ax.legend()
        _ = ax.set_xlabel(fig_data["xlabel"])
        _ = ax.set_ylabel(fig_data["ylabel"])
        y_lim = ax.get_ylim()
        _ = ax.set_ylim((y_lim[0], y_lim[1] * 1.1))

    figs.suptitle(fig_data["title"])

    return figs, axes

def draw_length(gt_lens, res_lens, Path):
    from statistics import mean
    print("min ans len in test split:  ", min(gt_lens))
    print("max ans len in test split:  ", max(gt_lens))
    print("mean ans len in test split: ", mean(gt_lens))
    print("min generated ans len:  ", min(res_lens))
    print("max generated ans len:  ", max(res_lens))
    print("mean generated ans len: ", mean(res_lens))

    title = "Histogram of answer length in test splits "
    label_x = "Length"
    label_y = "frequency"
    label_hist = ["Ground Truth", "Output"]
    fig_data = {
        "label_h": label_hist,
        "xlabel": label_x,
        "ylabel": label_y,
        "title": title
    }
    figs, axes = plot_hist(data=[gt_lens, res_lens],
                       fig_data=fig_data,
                       bins=[list(range(min(min(gt_lens), min(res_lens)), max(max(gt_lens),max(res_lens)), 2)),
                             list(range(min(min(gt_lens), min(res_lens)), max(max(gt_lens),max(res_lens)), 2))],
                       count=True)
    plt.savefig(Path)
    

def cnt_words(s):
    ss = s.split(" ")
    return len(ss)

def draw_bleu1(bleu1, Path):
    title = "Histogram and probaility normal distribution of bleu-1"
    label_x = "bleu-1"
    label_y = "frequency"
    label_hist = "bleu-1 histogram"
    label_line = "bleu-1 normal pdf"
    fig_data = {
        "label_h": [label_hist],
        "label_l": [label_line],
        "xlabel": label_x,
        "ylabel": label_y,
        "title": title
    }

    #bleu1 = bleu1.to_numpy()
    bins = [round(min(bleu1), 4), 0.5, 0.6, 0.75, 0.85, 0.95, round(max(bleu1), 4)]
    fig, ax = plot_hist(data=[bleu1],
                        fig_data=fig_data,
                        bins=[bins],
                        norm_pdf=True,
                        count=True)
    plt.savefig(Path)
