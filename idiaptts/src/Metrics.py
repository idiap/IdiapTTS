#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import math
import numpy as np
import logging
from functools import partial

import scipy
from nnmnkwii import metrics as nnmnkwii_metrics


class Metrics(object):

    # Available metrics.
    MCD = "MCD"
    F0_RMSE = "F0 RMSE"
    GPE = "GPE"
    FFE = "FFE"
    VDE = "VDE"
    BAP_distortion = "BAP distortion"
    Dur_RMSE = "Dur RMSE"
    Dur_pearson = "Dur pearson"

    _metric_formats = {MCD: "{:4.2f}dB",
                       F0_RMSE: "{:4.2f}Hz",
                       GPE: "{:2.2%}",
                       FFE: "{:2.2%}",
                       VDE: "{:2.2%}",
                       BAP_distortion: "{:4.2f}dB",
                       Dur_RMSE: "{:4.2f}Hz",
                       Dur_pearson: "{}"}

    _metric_format_functions = {Dur_pearson: partial(np.array_str, precision=2, suppress_small=True)}

    def __init__(self, metric_identifiers):

        self.metrics = metric_identifiers
        num_metrics = len(metric_identifiers)
        self.cum_values = [0] * num_metrics
        self.max_values = [None] * num_metrics
        self.max_value_ids = ["None"] * num_metrics
        self.cum_counts = [0] * num_metrics

    @staticmethod
    def get_metrics(metric_names,
                    org_coded_sp=None, org_lf0=None, org_vuv=None, org_bap=None,
                    output_coded_sp=None, output_lf0=None, output_vuv=None, output_bap=None,
                    org_dur=None, output_dur=None):

        metrics = []
        for metric_name in metric_names:
            if metric_name == Metrics.MCD:
                value = Metrics.mcd_k(org_coded_sp, output_coded_sp)
            elif metric_name.startswith(Metrics.MCD):
                mcd_k = int(metric_name.split('_')[1])
                value = Metrics.mcd_k(org_coded_sp, output_coded_sp, k=mcd_k)
            elif metric_name == Metrics.F0_RMSE:
                value = Metrics.f0_rmse(org_lf0, org_vuv, output_lf0)
            elif metric_name == Metrics.GPE:
                value = Metrics.gross_pitch_error(org_lf0, org_vuv, output_lf0, output_vuv)
            elif metric_name == Metrics.FFE:
                value = Metrics.f0_frame_error(org_lf0, org_vuv, output_lf0, output_vuv)
            elif metric_name == Metrics.VDE:
                value = Metrics.voicing_decision_error(org_vuv, output_vuv)
            elif metric_name == Metrics.BAP_distortion:
                value = Metrics.aperiodicity_distortion(org_bap, output_bap)
            elif metric_name == Metrics.Dur_RMSE:
                value = Metrics.rmse(org_dur, output_dur)
            elif metric_name == Metrics.Dur_pearson:
                value = Metrics.pearson(org_dur, output_dur)
            else:
                raise NotImplementedError("Unknown metric {}.".format(metric_name))

            metrics.append((metric_name, value))

        return metrics

    @staticmethod
    def mcd_k(org_cep, output_cep, k=None, start_bin=1):
        """Computes the Mel-cepstrum distortion of the first to k-th bin. Ignores c_0 (energy) by default."""
        org_coded_sp = org_cep[:len(output_cep)]
        if k is None:
            mcd = nnmnkwii_metrics.melcd(output_cep[:, start_bin:], org_coded_sp[:, start_bin:])  # TODO: Aligned mcd?
        else:
            mcd = nnmnkwii_metrics.melcd(output_cep[:, start_bin:k], org_coded_sp[:, start_bin:k])

        return mcd

    @staticmethod
    def f0_rmse(org_lf0, org_vuv, output_lf0):
        org_f0 = np.exp(org_lf0.squeeze())
        output_f0 = np.exp(output_lf0)

        org_f0 = org_f0[:len(output_lf0)]
        org_vuv = org_vuv[:len(output_lf0)]

        f0_mse = (org_f0 - output_f0) ** 2
        f0_rmse = math.sqrt((f0_mse * org_vuv).sum() / org_vuv.sum())

        return f0_rmse

    @staticmethod
    def gross_pitch_error(org_lf0, org_vuv, output_lf0, output_vuv):
        """
        "A method for fundamental frequency estimation and voicing decision:
        Application to infant utterances recorded in real acoustical environments"
        Nakatani, Tomohiro and Amano, Shigeaki and Irino, Toshio and Ishizuka, Kentaro and Kondo, Tadahisa
        https://www.sciencedirect.com/science/article/abs/pii/S0167639307001483

        The GPE measures the percentage of voiced frames that deviate in pitch by more than 20% compared to the ref.
        """
        org_lf0 = org_lf0[:len(output_lf0)]
        org_vuv = org_vuv[:len(output_lf0)]

        lf0_20perc_errors = np.abs(org_lf0 - output_lf0) > 0.2 * org_lf0
        both_voiced = org_vuv * output_vuv

        gpe = (lf0_20perc_errors * both_voiced).sum() / both_voiced.sum()

        return gpe

    @staticmethod
    def f0_frame_error(org_lf0, org_vuv, output_lf0, output_vuv):
        """
        "Reducing F0 frame error of f0 tracking algorithms under noisy conditions
         with an unvoiced/voiced classification frontend"
        Chu, Wei, and Abeer Alwan
        https://ieeexplore.ieee.org/abstract/document/4960497

        FFE measures the percentage of frames that either contain a 20% pitch error (according to GPE)
        or a voicing decision error (according to VDE).
        """
        org_lf0 = org_lf0[:len(output_lf0)]
        org_vuv = org_vuv[:len(output_lf0)]

        lf0_20perc_errors = np.abs(org_lf0 - output_lf0) > 0.2 * org_lf0
        both_voiced = org_vuv * output_vuv
        lf0_num_20perc_errors = (lf0_20perc_errors * both_voiced).sum()

        vde = Metrics.voicing_decision_error(org_vuv, output_vuv)

        return lf0_num_20perc_errors / len(output_vuv) + vde

    @staticmethod
    def voicing_decision_error(org_vuv, output_vuv):
        """VDE from the same paper as gross_pitch_error."""
        org_vuv = org_vuv[:len(output_vuv)]

        return (org_vuv != output_vuv).sum() / len(output_vuv)

    @staticmethod
    def aperiodicity_distortion(org_bap, output_bap):
        org_bap = org_bap[:len(output_bap)]

        if len(output_bap.shape) > 1 and output_bap.shape[1] > 1:
            return Metrics.mcd_k(org_bap, output_bap)
        else:
            return math.sqrt(((org_bap - output_bap) ** 2).mean()) * (10.0 / np.log(10) * np.sqrt(2.0))

    @staticmethod
    def rmse(org, output, axis=None):

        mse = (org - output) ** 2
        rmse = np.sqrt(mse.sum(axis=axis) / len(mse))

        return rmse

    @staticmethod
    def pearson(org, output):
        return np.array([scipy.stats.pearsonr(org[:, idx], output[:, idx])[0] for idx in range(org.shape[1])])

    def accumulate(self, id_name, current_metrics):

        for i, (metric_name, current) in enumerate(current_metrics):
            if np.isnan(current).any():
                logging.error("Computed NaN for {} for {}.".format(metric_name, id_name))
            else:
                if self.max_values[i] is None or np.array(current > self.max_values[i]).all():
                    self.max_values[i] = current
                    self.max_value_ids[i] = id_name
                self.cum_values[i] += current
                self.cum_counts[i] += 1

    def log(self):

        cum_values = self.get_cum_values()

        scores_formatted = []
        for i, metric_name in enumerate(self.metrics):
            metric_base_name = "MCD" if metric_name.startswith("MCD_") else metric_name
            metric_format = Metrics._metric_formats[metric_base_name]
            if metric_base_name in Metrics._metric_format_functions:
                format_fn: callable = Metrics._metric_format_functions[metric_base_name]
            else:
                format_fn: callable = lambda x: x  # No-op
            scores_formatted.append(("{} " + metric_format).format(metric_name,
                                                                   format_fn(cum_values[i])))
            logging.info(("Worst {}: {} " + metric_format).format(metric_name,
                                                                  self.max_value_ids[i],
                                                                  format_fn(self.max_values[i])))

        logging.info("Benchmark score: " + ", ".join(scores_formatted))

    def get_cum_values(self):
        return [cum_value / cum_count for cum_value, cum_count in zip(self.cum_values, self.cum_counts)]
