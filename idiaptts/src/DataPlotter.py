#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description.
Plot frame-wise data, areas, labels and atoms.
"""

# System imports.
import os
import colorsys
import logging
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from itertools import cycle

# Third-party imports.
import librosa.display

# Local source tree imports.
from idiaptts.misc.utils import makedirs_safe


class DataPlotter(object):
    """Class description.
    """
    logger = logging.getLogger(__name__)

    # Constants.
    FILE_EXTENSION = ".png"

    class Grid(object):
        def __init__(self):
            self.data_list = None
            self.atom_list = None
            self.area_list = None
            self.spec = None
            self.annotations = None
            self.xmin = None
            self.xmax = None
            self.ymin = None
            self.ymax = None
            self.xlabel = None
            self.ylabel = None
            self.linestyles = None
            self.hatchstyles = None
            self.colors = None
            self.alpha = None
            self.linewidth = None

    def __init__(self):
        self.logger = self.logger or logging.getLogger(__name__)
        self.debug = False

        self.grids = list()
        self.plt = None
        self.title = None
        self.num_colors = None

    def set_data_list(self, grid_idx, data_list):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].data_list = data_list

    def set_atom_list(self, grid_idx, atom_list):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].atom_list = atom_list

    def set_area_list(self, grid_idx, area_list):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].area_list = area_list

    def set_specshow(self, grid_idx, spec):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].spec = spec

    def set_annotations(self, grid_idx, annotations):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].annotations = annotations

    def set_hatchstyles(self, grid_idx, hatchstyles):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].hatchstyles = cycle(hatchstyles)

    def set_linestyles(self, grid_idx, linestyles):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].linestyles = linestyles

    def set_colors(self, grid_idx, colors=None, alpha=None):
        self._create_missing_grids(grid_idx)
        if colors is not None:
            self.grids[grid_idx].colors = cycle(colors)
        if alpha is not None:
            self.grids[grid_idx].alpha = alpha

    def set_linewidth(self, grid_idx, linewidth):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].linewidth = linewidth

    def set_lim(self, grid_idx=None, xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Set lim for the plots. If None automatic lim from matplotlib is used.
        If grid_idx is None, lim is set for all already known grids meaning data has to be loaded first!
        grid_idx can be an integer or a list of integers.
        """
        if grid_idx is None:
            if len(self.grids) == 0:
                self.logger.warning("No grids are available, change of lim has no effect. Add grids first or specify grid_idx.")
            for g in self.grids:
                g.xmin = xmin if xmin is not None else g.xmin
                g.xmax = xmax if xmax is not None else g.xmax
                g.ymin = ymin if ymin is not None else g.ymin
                g.ymax = ymax if ymax is not None else g.ymax
        else:
            if isinstance(grid_idx, collections.Sequence):
                for idx in grid_idx:
                    idx = int(idx)
                    self._create_missing_grids(idx)
                    self.grids[idx].xmin = xmin if xmin is not None else self.grids[idx].xmin
                    self.grids[idx].xmax = xmax if xmax is not None else self.grids[idx].xmax
                    self.grids[idx].ymin = ymin if ymin is not None else self.grids[idx].ymin
                    self.grids[idx].ymax = ymax if ymax is not None else self.grids[idx].ymax
            else:
                grid_idx = int(grid_idx)
                self._create_missing_grids(grid_idx)
                self.grids[grid_idx].xmin = xmin if xmin is not None else self.grids[grid_idx].xmin
                self.grids[grid_idx].xmax = xmax if xmax is not None else self.grids[grid_idx].xmax
                self.grids[grid_idx].ymin = ymin if ymin is not None else self.grids[grid_idx].ymin
                self.grids[grid_idx].ymax = ymax if ymax is not None else self.grids[grid_idx].ymax

    # def get_xlim(self, grid_idx):
    #     if len(self.grids) < grid_idx + 1:
    #         self.logger.error("Trying to get lim of non-existing grid index: " + str(grid_idx))
    #         return None
    #     return grid_idx.ax.get_xlim()
    #
    # def get_ylim(self, grid_idx):
    #     if len(self.grids) < grid_idx + 1:
    #         self.logger.error("Trying to get lim of non-existing grid index: " + str(grid_idx))
    #         return None
    #     return grid_idx.ax.get_ylim()

    def set_label(self, grid_idx=None, xlabel=None, ylabel=None):
        if grid_idx is None:
            if len(self.grids) == 0:
                self.logger.warning(
                    "No grids are available, change of label has no effect. Add grids first or specify grid_idx.")
            for g in self.grids:
                g.xlabel = xlabel
                g.ylabel = ylabel
        else:
            if isinstance(grid_idx, collections.Sequence):
                for idx in grid_idx:
                    idx = int(idx)
                    self._create_missing_grids(idx)
                    self.grids[idx].xlabel = xlabel
                    self.grids[idx].ylabel = ylabel
            else:
                grid_idx = int(grid_idx)
                self._create_missing_grids(grid_idx)
                self.grids[grid_idx].xlabel = xlabel
                self.grids[grid_idx].ylabel = ylabel

    def set_title(self, title):
        self.title = title

    def save_to_file(self, filename):
        if self.plt is None:
            logging.error("No generated plot exists, please run 'gen_plot()' first.")
        else:
            makedirs_safe(os.path.dirname(filename))
            self.plt.savefig(filename, bbox_inches=0)
            logging.info("Figure saved as " + filename)

    def set_num_colors(self, num):
        """If num is None, colors are computed by number of data graphs."""
        self.num_colors = num

    @staticmethod
    def get_color(ncolor, monochrome=False):
        """
        Returns ncolor colors
        """
        if monochrome:
            for value in cycle(range(ncolor)):
                value = 1. * value / ncolor
                col = [int(x) for x in colorsys.hsv_to_rgb(0, 0, value*255*0.6)]
                yield '#{0:02x}{1:02x}{2:02x}'.format(*col)

        for hue in cycle(range(ncolor)):
            hue = 1. * hue / ncolor
            col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
            yield '#{0:02x}{1:02x}{2:02x}'.format(*col)

    def _create_missing_grids(self, requested_grid_idx):
        # Add grids until index is valid.
        while len(self.grids) <= requested_grid_idx:
            self.grids.append(self.Grid())

    def gen_plot(self, monochrome=False, sharex=False):
        if not sharex:
            fig = plt.figure(figsize=(12.75, 8))
            gs = gridspec.GridSpec(len(self.grids), 1)
        else:
            fig, gs = plt.subplots(len(self.grids), 1, sharex=True, figsize=(12.75, 8))

        max_length = 0.0
        for grid in self.grids:
            if grid.data_list is not None:
                for data in grid.data_list:
                    max_length = max(max_length, len(data[0]))
            # elif grid.atom_list is not None:
            #     for atom in grid.atom_list:
            #         max_length = max(max_length, atom.position + atom.length + 10)
        t = np.arange(0, max_length)

        for grid_idx, grid in reversed(list(enumerate(self.grids))):  # Reversed order so that legend is not overlaid.
            grid.ax = plt.subplot(gs[grid_idx])

            legend_plots = []
            legend_names = []
            num_graphs = len(grid.data_list) if grid.data_list is not None else 0
            num_graphs += len(grid.atom_list) if grid.atom_list is not None else 0
            color = DataPlotter.get_color(num_graphs, monochrome) if self.num_colors is None else DataPlotter.get_color(self.num_colors, monochrome)
            alpha = 1 if monochrome else 0.8
            if grid.colors is not None and not monochrome:
                    color = grid.colors

            if grid.alpha is not None and not monochrome:
                    alpha = grid.alpha

            if grid.data_list is not None:
                linestyles = ['-'] if grid.linestyles is None else grid.linestyles
                linewidth = [1.5] if grid.linewidth is None else grid.linewidth
                line_num = 0
                for data in grid.data_list:
                    name = None if len(data) == 1 else data[1]
                    data = data[0]

                    acolor = next(color)
                    if data.ndim > 1 and data.shape[1] > 1:
                        self.logger.error((str(name) + " has" if name is not None else "Data has") + " too many dimensions (" + str(data.shape) + ").")
                        continue
                    plot, = plt.plot(t[:len(data)], data, acolor, linestyle=linestyles[line_num % len(linestyles)], linewidth=linewidth[line_num % len(linewidth)], alpha=alpha)
                    line_num += 1
                    if name is not None:
                        legend_plots.append(plot)
                        legend_names.append(name)
            if grid.atom_list is not None:
                spikes = np.zeros((max_length))
                for atom in grid.atom_list:
                    acolor = next(color)
                    spikes[atom.position] = atom.amp
                    plt.plot(t, atom.get_padded_curve(max_length), color=acolor, linewidth=1.5)
            if grid.area_list is not None:
                hatches = cycle(['//', '\\\\', '--', '||'])
                if grid.hatchstyles is not None:
                    hatches = grid.hatchstyles

                for area in grid.area_list:
                    hatch = next(hatches)
                    area_flags = area[0]
                    area_color = '0.8' if len(area) < 2 else area[1]
                    area_alpha = 1.0 if len(area) < 3 else area[2]
                    x1 = 0
                    x2 = x1 + 1
                    zone = None
                    while True:
                        # Case 1: x1 is out of area continue until x2 reaches an area or end point.
                        if area_flags[x1] == 0:
                            if x2 == len(area_flags):
                                break
                            if area_flags[x2] != 0:
                                x1 = x2
                        else:
                            if x2 == len(area_flags):
                                zone = plt.axvspan(x1, x2 - 1, color=area_color, alpha=area_alpha, hatch=hatch, fill=False)
                                break
                            if area_flags[x2] == 0:
                                plt.axvspan(x1, x2 - 1, color=area_color, alpha=area_alpha, hatch=hatch, fill=False)
                                x1 = x2
                        x2 = x2 + 1
                    if len(area) > 3 and zone is not None:
                        legend_names.append(area[3])
                        legend_plots.append(zone)

            if grid.spec is not None:
                librosa.display.specshow(grid.spec.T)
                # plt.colorbar(format='%+2.0f DB')

            if grid.xlabel is not None:
                plt.xlabel(grid.xlabel)
            if grid.ylabel is not None:
                plt.ylabel(grid.ylabel)

            plt.xlim(0.0, max_length)  # Default
            plt.xlim(grid.xmin, grid.xmax)  # None values have no effect.
            plt.ylim(grid.ymin, grid.ymax)

            if plt.ylim()[0] <= 0 and plt.ylim()[1] >= 0:
                plt.axhline(y=0, color='k', linestyle=':', linewidth=1)

            if grid.annotations is not None:
                x_axis_twin = grid.ax.twiny()
                x_axis_twin.xaxis.set_major_locator(ticker.FixedLocator(grid.annotations[:, 0].astype(np.float64), nbins=len(grid.annotations[:, 0]) + 1))
                grid.annotations[1::2, 1] = np.core.defchararray.add(grid.annotations[1::2, 1], '\n')
                x_axis_twin.xaxis.set_major_formatter(ticker.FixedFormatter(grid.annotations[:, 1]))
                x_axis_twin.set_xlim(0.0, max_length)
                x_axis_twin.set_xlim(grid.xmin, grid.xmax)
                for tick in x_axis_twin.xaxis.get_majorticklabels():
                    tick.set_horizontalalignment("left")
                    tick.set_fontsize(8)

            if len(legend_plots) > 0:
                plt.legend(legend_plots, legend_names, ncol=len(legend_names), fontsize=10, frameon=False,
                           bbox_to_anchor=(0., 0.98, 0., .102), loc=3, borderaxespad=None if grid.annotations is None else 2.5)

        if self.title is not None:
            plt.sca(self.grids[0].ax)
            plt.title(self.title, size=11, y=1.4)
            # plt.subplot(gs[0])
            #fig.suptitle(self.title, size=11)
        for i in range(5):
            plt.tight_layout()
        if sharex:
            plt.subplots_adjust(hspace=0.35)
        else:
            plt.subplots_adjust(hspace=0.5)
        plt.show()
        self.plt = plt
