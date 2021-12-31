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
import collections
import colorsys
import contextlib
import copy
from io import UnsupportedOperation
from itertools import cycle
import logging
import os
from typing import Callable, ContextManager, List, Tuple, Union

# Third-party imports.
import matplotlib
if bool(os.environ.get('DISPLAY', None)) and not os.environ.get('TEST_FLAG'):
    matplotlib.use('TkAgg')
    has_display = True
else:
    matplotlib.use('Agg')
    has_display = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.cm import get_cmap
import numpy as np

# Local source tree imports.
from idiaptts.misc.utils import makedirs_safe


class DataPlotter(object):
    """Class description.
    """
    logger = logging.getLogger(__name__)

    class Config:
        def __init__(self,
                     feature_name: str,
                     plot_fn: Callable,
                     post_processed: bool,
                     annotation_fn: Callable = None,
                     plotter_name: str = "default",
                     grid_indices: List[int] = None) -> None:
            self.feature_name = feature_name
            self.plot_fn = plot_fn
            self.post_processed = post_processed
            self.annotation_fn = annotation_fn
            self.plotter_name = plotter_name
            self.grid_indices = grid_indices

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
            self.hlines = list()

    def __init__(self):
        self.logger = self.logger or logging.getLogger(__name__)
        self.debug = False

        self.grids = list()
        self.plt = None
        self.title = None
        self.num_colors = None

        self.inside_context = False

    def __enter__(self):
        self.inside_context = True
        return self

    def __exit__(self, *exc):
        if self.plt is not None:
            self.logger.debug("Close figure.")
            plt.close()
        self.inside_context = False

    def get_next_free_grid_idx(self):
        return len(self.grids)

    def get_all_grid_indices(self):
        return range(len(self.grids))

    def set_data_list(self, grid_idx: int, data_list: Tuple):
        """
        Set data to plot. Data has to be 1d or size 1 in second dimension.
        The data can be given as (data, legend name, x_coords) where legend name and x_coords are optional.
        """
        self._create_missing_grids(grid_idx)
        if self.grids[grid_idx].data_list is None:
            self.grids[grid_idx].data_list = data_list
        else:
            self.grids[grid_idx].data_list += data_list

    def set_atom_list(self, grid_idx: int, atom_list: List):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].atom_list = atom_list

    def set_area_list(self, grid_idx: int, area_list: Tuple):
        self._create_missing_grids(grid_idx)
        if self.grids[grid_idx].area_list is None:
            self.grids[grid_idx].area_list = area_list
        else:
            self.grids[grid_idx].area_list += area_list

    def set_specshow(self, grid_idx: int, spec: np.ndarray):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].spec = spec

    def set_annotations(self, grid_idx: int, annotations: np.ndarray):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].annotations = annotations

    def set_hatchstyles(self, grid_idx: int, hatchstyles: List[str]):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].hatchstyles = cycle(hatchstyles)

    def set_linestyles(self, grid_idx: int, linestyles: List[str]):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].linestyles = linestyles

    def set_colors(self, grid_idx: int, colors: Union[str, List[str]] = None,
                   alpha: Union[float, List[float]] = None):
        self._create_missing_grids(grid_idx)
        if colors is not None:
            self.grids[grid_idx].colors = cycle(colors)
        if alpha is not None:
            self.grids[grid_idx].alpha = alpha

    def set_linewidth(self, grid_idx: int, linewidth: List[float]):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].linewidth = linewidth

    def add_hline(self, grid_idx, y, xmin=0, xmax=1, kwargs=None):
        self._create_missing_grids(grid_idx)
        self.grids[grid_idx].hlines += [(y, xmin, xmax, kwargs)]

    def set_lim(self, grid_idx=None, xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Set lim for the plots. If None automatic lim from matplotlib is used.
        If grid_idx is None, lim is set for all already known grids meaning data has to be loaded first!
        grid_idx can be an integer or a list of integers.
        """
        if grid_idx is None:
            if len(self.grids) == 0:
                self.logger.warning("No grids are available, change of lim has'\
                                    ' no effect. Add grids first or specify '\
                                    'grid_idx.")
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
                self.logger.warning("No grids available, change of label has no"
                                    " effect. Add grids or specify grid_idx.")
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
            self.plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            logging.info("Figure saved as " + filename)

    def set_num_colors(self, num):
        """If num is None, colors are computed by number of data graphs."""
        self.num_colors = num

    def _create_missing_grids(self, requested_grid_idx):
        # Add grids until index is valid.
        while len(self.grids) <= requested_grid_idx:
            self.grids.append(self.Grid())

    def gen_plot(self, monochrome=False, sharex=False, legend_kwargs={}):
        """Main function to create the plot."""
        if not self.inside_context:
            raise NotImplementedError(
                "Please create the {} class as context manager.".format(
                    type(self).__name__))
        if self.plt is not None:
            # Method was called before, close current figure.
            plt.close()

        fig, gs = self._create_figure_and_grid(sharex)

        max_length = self._find_max_data_length()
        t = np.arange(0, max_length)

        for grid_idx, grid in reversed(list(enumerate(self.grids))):  # Reversed order so that legend is not overlaid.
            self._gen_grid_plot(gs, t, grid_idx, grid, monochrome, legend_kwargs)

        if self.title is not None:
            plt.sca(self.grids[0].ax)
            plt.title(self.title, size=11, y=1.4)
            # plt.subplot(gs[0])
            # fig.suptitle(self.title, size=11)

        if sharex:
            hspace = 0.35
        else:
            hspace = 0.5
        if any(map(lambda g: g.annotations is not None, self.grids)):
            hspace += 0.3
        plt.subplots_adjust(hspace=hspace)

        for _ in range(5):
            plt.tight_layout()

        if has_display:
            plt.show()
        self.plt = plt
        return fig

    def _create_figure_and_grid(self, sharex):
        if not sharex:
            fig = plt.figure(figsize=(12.75, 3 * len(self.grids)))
            gs = gridspec.GridSpec(len(self.grids), 1)
        else:
            fig, gs = plt.subplots(len(self.grids), 1, sharex=True,
                                   figsize=(12.75, 8))

        return fig, gs

    def _find_max_data_length(self):
        max_length = 0.0
        for grid in self.grids:
            if grid.data_list is not None:
                for data in grid.data_list:
                    if len(data) == 3:  # If data has three elements it is (data, name, indices).
                        max_length = max(max_length, data[2][-1])
                    else:
                        max_length = max(max_length, len(data[0]))
            # elif grid.atom_list is not None:
            #     for atom in grid.atom_list:
            #         max_length = max(max_length, atom.position + atom.length + 10)
            if grid.spec is not None:
                max_length = max(max_length, len(grid.spec))

        return max_length

    def _gen_grid_plot(self, gs, t, grid_idx, grid, monochrome, legend_kwargs):
        grid.ax = plt.subplot(gs[grid_idx])
        max_length = len(t)

        legend_plots = []
        legend_names = []
        num_graphs = len(grid.data_list) if grid.data_list is not None else 0
        if grid.atom_list is not None:
            num_graphs += len(grid.atom_list)

        color = self._get_colors(grid, num_graphs, monochrome)
        alpha = self._get_alpha(grid, monochrome)

        if grid.data_list is not None:
            self._plot_data_list(grid, t, color, alpha, legend_plots,
                                 legend_names)

        if grid.atom_list is not None:
            self._plot_atom_list(grid, max_length, t, color)
        if grid.area_list is not None:
            self._plot_area_list(grid, legend_plots, legend_names)

        if grid.spec is not None:
            self._plot_spec(grid)

        # Add horizontal lines.
        if len(grid.hlines) > 0:
            for y, xmin, xmax, kwargs in grid.hlines:
                if kwargs is not None:
                    grid.ax.axhline(y, xmin, xmax, **kwargs)
                else:
                    grid.ax.axhline(y, xmin, xmax)

        if grid.xlabel is not None:
            plt.xlabel(grid.xlabel, fontsize=12)
        if grid.ylabel is not None:
            plt.ylabel(grid.ylabel, fontsize=12)

        plt.xlim(0.0, max_length)
        plt.xlim(grid.xmin, grid.xmax)  # None values have no effect.
        plt.ylim(grid.ymin, grid.ymax)

        if plt.ylim()[0] <= 0 and plt.ylim()[1] >= 0:
            plt.axhline(y=0, color='k', linestyle=':', linewidth=1)

        if grid.annotations is not None:
            self._plot_annotation(grid, max_length)

        if len(legend_plots) > 0:
            self._plot_legend(legend_names, legend_plots,
                              has_annotation=grid.annotations is not None,
                              legend_kwargs=legend_kwargs)

    def _get_colors(self, grid, num_graphs, monochrome):
        if grid.colors is not None and not monochrome:
            color = grid.colors
        elif self.num_colors is None:
            color = DataPlotter.get_color(num_graphs, monochrome)
        else:
            color = DataPlotter.get_color(self.num_colors, monochrome)

        return color

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

    def _get_alpha(self, grid, monochrome):
        if grid.alpha is not None and not monochrome:
            alpha = grid.alpha
        else:
            alpha = 1 if monochrome else 0.8
        return alpha

    def _plot_data_list(self, grid, t, color, alpha, legend_plots, legend_names):
        linestyles = ['-'] if grid.linestyles is None else grid.linestyles
        linewidth = [1.5] if grid.linewidth is None else grid.linewidth
        line_num = 0
        for data in grid.data_list:
            name = None if len(data) == 1 else data[1]
            x_coords = data[2] if len(data) == 3 else t[:len(data[0])]
            data = data[0]

            acolor = next(color)
            if data.ndim > 1 and data.shape[1] > 1:
                self.logger.error("{} has too many dimension ({}).".format(
                    name if name is not None else "Data", data.shape))
                continue
            plot, = plt.plot(x_coords,
                             data,
                             acolor,
                             linestyle=linestyles[line_num % len(linestyles)],
                             linewidth=linewidth[line_num % len(linewidth)],
                             alpha=alpha)
            line_num += 1
            if name is not None:
                legend_plots.append(plot)
                legend_names.append(name)

    def _plot_atom_list(self, grid, max_length, t, color):
        # from tools.wcad.wcad.object_types.atom import Atom
        spikes = np.zeros((max_length))
        for atom in grid.atom_list:
            next_color = next(color)
            spikes[atom.position] = atom.amp
            plt.plot(t, atom.get_padded_curve(max_length), color=next_color,
                     linewidth=1.5)

    def _plot_area_list(self, grid, legend_plots, legend_names):
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
                        zone = plt.axvspan(x1, x2 - 1, color=area_color,
                                           alpha=area_alpha, hatch=hatch,
                                           fill=False)
                        break
                    if area_flags[x2] == 0:
                        plt.axvspan(x1, x2 - 1, color=area_color,
                                    alpha=area_alpha, hatch=hatch, fill=False)
                        x1 = x2
                x2 = x2 + 1
            if len(area) > 3 and zone is not None:
                legend_names.append(area[3])
                legend_plots.append(zone)

    def _plot_spec(self, grid):
        if np.issubdtype(grid.spec.dtype, np.complexfloating):
            logging.warning('Trying to display complex-valued input. '
                            'Showing magnitude instead.')
            grid.spec = np.absolute(grid.spec)
        grid.ax.yaxis.set_major_locator(plt.NullLocator())
        grid.ax.xaxis.set_minor_locator(plt.NullLocator())
        plt.pcolormesh(grid.spec.T, cmap=cmap(grid.spec.T), rasterized=True,
                       edgecolor='None', shading='flat')
        # import librosa.display
        # librosa.display.specshow(grid.spec.T)
        # plt.colorbar(format='%+2.0f DB')

    def _plot_annotation(self, grid, max_length):
            x_axis_twin = grid.ax.twiny()  # Get x-axis above plot.

            # Set ticks for all annotation.
            x_axis_twin.xaxis.set_major_locator(ticker.FixedLocator(
                grid.annotations[:, 0].astype(np.float64),
                nbins=len(grid.annotations[:, 0]) + 1))

            annotations = copy.deepcopy(grid.annotations)
            # Shift every other annotation up.
            annotations[1::2, 1] = np.core.defchararray.add(
                grid.annotations[1::2, 1], '\n')
            x_axis_twin.xaxis.set_major_formatter(ticker.FixedFormatter(
                annotations[:, 1]))
            x_axis_twin.set_xlim(0.0, max_length)
            x_axis_twin.set_xlim(grid.xmin, grid.xmax)

            for tick in x_axis_twin.xaxis.get_majorticklabels():
                tick.set_horizontalalignment("left")
                tick.set_fontsize(10)

    def _plot_legend(self, legend_names, legend_plots, has_annotation, legend_kwargs):
        default_legend_kwargs = {
            "ncol": len(legend_names),
            "fontsize": 12,
            "frameon": False,
            "bbox_to_anchor": (0., 0.98, 0., .102),
            "loc": 3,
            "borderaxespad": 2.5 if has_annotation else None
        }
        default_legend_kwargs.update(legend_kwargs)
        # # Insert dummy entries in the legend to manipulate the alignment for multi-column legends.
        # legend_plots.insert(2, plt.plot([], [], color=(0, 0, 0, 0), label=" ")[0])
        # legend_names.insert(2, '')
        plt.legend(legend_plots, legend_names, **default_legend_kwargs)


def cmap(data, robust=True, cmap_seq='magma', cmap_bool='gray_r', cmap_div='coolwarm'):
    """
    Copyright (c) 2013--2017, librosa development team.
    Librosa cmap.
    """

    data = np.atleast_1d(data)

    if data.dtype == 'bool':
        return get_cmap(cmap_bool)

    data = data[np.isfinite(data)]

    if robust:
        min_p, max_p = 2, 98
    else:
        min_p, max_p = 0, 100

    max_val = np.percentile(data, max_p)
    min_val = np.percentile(data, min_p)

    if min_val >= 0 or max_val <= 0:
        return get_cmap(cmap_seq)

    return get_cmap(cmap_div)
