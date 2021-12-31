#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Train a model to predict mgc, lf0 and bap with deltas and double
   deltas and synthesize audio from it by using MLPG for smoothing.
"""

# System imports.
import copy
from functools import partial
import logging
import numpy as np
import os
from typing import Dict, List, Tuple, Union

# Third-party imports.

# Local source tree imports.
from idiaptts.src.model_trainers.ModularTrainer import ModularTrainer
from idiaptts.src.DataPlotter import DataPlotter
from idiaptts.src.Metrics import Metrics
from idiaptts.src.data_preparation.DataReaderConfig import _get_padding_sizes
from idiaptts.src.ExtendedHParams import ExtendedHParams
from idiaptts.src.data_preparation.audio.AudioProcessing import AudioProcessing
from idiaptts.src.data_preparation.NpzDataReader import DataReader, NpzDataReader
from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import interpolate_lin
from idiaptts.src.neural_networks.pytorch.loss.NamedLoss import NamedLoss
from idiaptts.src.neural_networks.pytorch.models.NamedForwardWrapper import NamedForwardWrapper
import idiaptts.src.neural_networks.pytorch.models.rnn_dyn as rnn_dyn


class AcousticModelTrainer(ModularTrainer):
    """
    Implementation of a ModularTrainer for the generation of acoustic
    data. Use question labels as input and WORLD features w/ or w/o
    deltas & double deltas (specified in hparams.add_deltas) as output.
    Synthesize audio from model output with MLPG smoothing.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, hparams: ExtendedHParams, id_list: List[str],
                 data_reader_configs: List[DataReader.Config] = None):

        if hparams is None:
            hparams = self.create_hparams()
            hparams.out_dir = os.path.curdir

        super(AcousticModelTrainer, self).__init__(
            data_reader_configs=data_reader_configs,
            id_list=id_list,
            hparams=hparams)

        if hparams.scheduler_type == "default":
            hparams.scheduler_type = "Plateau"
            hparams.add_hparams(plateau_verbose=True)

    @staticmethod
    def legacy_support_init(dir_world_features: os.PathLike,
                            dir_question_labels: os.PathLike,
                            id_list: List[str],
                            num_questions: int,
                            hparams: ExtendedHParams):
        """Get arguments for new init.

        :param dir_world_features:   Path to the directory containing
                                     the world features.
        :param dir_question_labels:  Path to the directory containing
                                     the question labels.
        :param id_list:              List of ids, can contain a speaker
                                     directory.
        :param num_questions:        Number of questions in question
                                     file (only needed for legacy code).
        :param hparams:              Set of hyper parameters.
        """
        data_reader_configs = []
        from idiaptts.src.data_preparation.DataReaderConfig import DataReaderConfig
        data_reader_configs.append(
            DataReaderConfig(
                name="questions",
                feature_type="QuestionLabelGen",
                directory=dir_question_labels,
                features="questions",
                num_questions=num_questions,
                match_length=["cmp_features"]
            )
        )
        # if hasattr(hparams, "add_deltas") and hparams.add_deltas:
        data_reader_configs.append(
            WorldFeatLabelGen.Config(
                name="cmp_features",
                # feature_type="WorldFeatLabelGen",
                directory=dir_world_features,
                features=["cmp_mcep" + str(hparams.num_coded_sps)],
                output_names=["acoustic_features"],
                add_deltas=hparams.add_deltas,
                num_coded_sps=hparams.num_coded_sps,
                num_bap=hparams.num_bap,
                sp_type=hparams.sp_type,
                requires_seq_mask=True,
                match_length=["questions"]
            )
        )
        hparams.world_dir = dir_world_features
        # else:
            # # TODO: How to load them separately?
            # datareader_configs.append(
            #     DataReader.Config(
            #         name="cmp_features",
            #         feature_type="WorldFeatLabelGen",
            #         directory=dir_world_features,
            #         features=["cmp_mcep" + str(hparams.num_coded_sps)],
            #         output_names=["acoustic_features"],
            #         add_deltas=hparams.add_deltas,
            #         num_coded_sps=hparams.num_coded_sps,
            #         num_bap=hparams.num_bap,
            #         sp_type=hparams.sp_type,
            #         requires_seq_mask=True
            #     )
            # )

        return dict(data_reader_configs=data_reader_configs, hparams=hparams,
                    id_list=id_list)

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        """
        Create model hyper parameter container. Parse non default from
        given string.
        """
        hparams = ModularTrainer.create_hparams(hparams_string, verbose=False)

        hparams.add_hparams(
            num_questions=None,
            question_file=None,  # Used to add labels in plot.
            num_coded_sps=60,
            num_baps=1,
            load_sp=True,
            load_lf0=True,
            load_vuv=True,
            load_bap=True,
            sp_type="mcep",
            add_deltas=True,
            synth_load_org_sp=False,
            synth_load_org_lf0=False,
            synth_load_org_vuv=False,
            synth_load_org_bap=False,
            # More available metrics in the Metrics class.
            metrics=[Metrics.MCD, Metrics.F0_RMSE, Metrics.VDE,
                     Metrics.BAP_distortion])

        if verbose:
            logging.info(hparams.get_debug_string())

        return hparams

    def init(self,
             hparams: ExtendedHParams,
             data_reader_configs: List[DataReader.Config] = None,
             model_config=None,
             loss_configs: List[NamedLoss.Config] = None) -> None:

        if model_config is None and hparams.has_value("model_type"):
            model_config = NamedForwardWrapper.Config(
                wrapped_model_config=rnn_dyn.convert_legacy_to_config(
                    (hparams.num_questions,), hparams),
                input_names="questions",
                batch_first=hparams.batch_first,
                name="AcousticModel",
                output_names="pred_acoustic_features"
            )

        if loss_configs is None:
            loss_configs = [NamedLoss.Config(
                name="MSELoss_acoustic_features",
                type_="MSELoss",
                seq_mask="acoustic_features_mask",
                input_names=["acoustic_features", "pred_acoustic_features"],
                batch_first=hparams.batch_first)]

        super().init(data_reader_configs=data_reader_configs, hparams=hparams,
                     model_config=model_config, loss_configs=loss_configs)
        self.logger.info("AcousticModelTrainer ready.")

    @staticmethod
    def plot_world_features(
            plotter: DataPlotter,
            plotter_config: DataPlotter.Config,
            grid_indices: List[int],
            id_name: str,
            features: np.ndarray,
            contains_deltas: bool,
            num_coded_sps: int,
            num_bap: int,
            hparams: ExtendedHParams,
            plot_mgc: bool = True,
            mgc_label: str = "Original spectrogram",
            plot_lf0: bool = True,
            sps_slices: slice = None,
            lf0_label: str = "Original LF0",
            plot_vuv: bool = True,
            vuv_colour_alpha: List[Union[str, float]] = ('red', 0.2),
            *args,
            **kwargs):

        mgc, lf0, vuv, _ = WorldFeatLabelGen.convert_to_world_features(
            features, contains_deltas=contains_deltas,
            num_coded_sps=num_coded_sps, num_bap=num_bap)
        lf0, _ = interpolate_lin(lf0)

        if grid_indices is None:
            grid_idx = plotter.get_next_free_grid_idx()
        else:
            grid_idx = grid_indices[0]

        if plot_lf0:

            plotter.set_label(
                grid_idx=grid_idx,
                xlabel='frames [{} ms]'.format(hparams.frame_size_ms),
                ylabel='log(f0)')
            # plotter.set_lim(grid_idx=0, ymin=math.log(60), ymax=math.log(250))
            plotter.set_data_list(grid_idx=grid_idx,
                                  data_list=[(lf0, lf0_label)])

        if plot_vuv:
            plotter.set_area_list(
                grid_idx=grid_idx,
                area_list=[(np.invert(vuv.astype(bool)), *vuv_colour_alpha)])

        if plot_mgc:
            if grid_indices is None:
                grid_idx = plotter.get_next_free_grid_idx()
            else:
                grid_idx = grid_indices[1]

            AcousticModelTrainer.plot_mgc(
                plotter=plotter,
                plotter_config=plotter_config,
                grid_indices=[grid_idx],
                id_name=id_name,
                features=mgc,
                synth_fs=hparams.synth_fs,
                spec_slice=sps_slices,
                labels=('frames [{} ms]'.format(hparams.frame_size_ms),
                        mgc_label),
                *args,
                **kwargs)

    @staticmethod
    def plot_mgc(
            plotter: DataPlotter,
            plotter_config: DataPlotter.Config,
            grid_indices: List[int],
            id_name: str,
            features: np.ndarray,
            synth_fs: int,
            spec_slice: slice = None,
            labels: Tuple[str, str] = (None, None),
            xlim: Union[str, Tuple[float, float]] = (None, None),
            ylim: Union[str, Tuple[float, float]] = (None, None),
            *args,
            **kwargs):

        import librosa
        amp_sp = np.absolute(AudioProcessing.mcep_to_amp_sp(
            features, synth_fs))
        amp_sp_db =librosa.amplitude_to_db(amp_sp, top_db=None)

        ModularTrainer.plot_specshow(plotter, plotter_config, grid_indices,
                                     id_name, amp_sp_db, spec_slice, labels,
                                     xlim, ylim, *args, **kwargs)

    @staticmethod
    def plot_phoneme_annotations(plotter: DataPlotter, id_name: str,
                                 hparams: ExtendedHParams,
                                 num_questions: int,
                                 phoneme_indices: np.ndarray,
                                 question_dir: os.PathLike,
                                 question_file: os.PathLike,
                                 grid_indices: List[int] = None):

        questions = QuestionLabelGen.load_sample(
            id_name=id_name,
            dir_out=question_dir,
            num_questions=num_questions)
        np_phonemes = QuestionLabelGen.questions_to_phonemes(
            questions,
            phoneme_indices,
            question_file)

        if grid_indices is None:
            grid_indices = plotter.get_all_grid_indices()
        for grid_idx in grid_indices:
            plotter.set_annotations(grid_idx, np_phonemes)

    def benchmark(
            self, hparams: ExtendedHParams,
            post_processing_mapping: Dict[str, str] = None,
            ids_input: Union[str, List[str], Tuple[str, ...], os.PathLike] = None):

        if post_processing_mapping is None:
            post_processing_mapping = {"pred_acoustic_features": "cmp_features"}

        return super().benchmark(
            hparams=hparams,
            post_processing_mapping=post_processing_mapping,
            ids_input=ids_input)

    def gen_figure(
            self,
            hparams: ExtendedHParams,
            ids_input: Union[str, List[str], Tuple[str, ...], os.PathLike],
            post_processing_mapping: Dict[str, str] = None,
            plotter_configs: List[DataPlotter.Config] = None):

        if post_processing_mapping is None:
            post_processing_mapping = {"pred_acoustic_features": "cmp_features",
                                       "acoustic_features": "cmp_features"}

        if plotter_configs is None:
            plotter_configs = AcousticModelTrainer._get_legacy_plotter_configs(
                hparams)

        return super().gen_figure(
            hparams=hparams,
            ids_input=ids_input,
            post_processing_mapping=post_processing_mapping,
            plotter_configs=plotter_configs)

    @staticmethod
    def _get_legacy_plotter_configs(hparams):
        if hparams.has_value("phoneme_indices") \
                and hparams.has_value("question_file") \
                and hparams.has_value("num_questions"):
            annotation_fn = partial(
                AcousticModelTrainer.plot_phoneme_annotations,
                num_questions=hparams.num_questions,
                phoneme_indices=hparams.phoneme_indices,
                question_file=hparams.question_file,
                question_dir=os.path.join(hparams.work_dir, "questions")
            )
        else:
            annotation_fn = None

        return [
            DataPlotter.Config(
                annotation_fn=annotation_fn,
                feature_name="pred_acoustic_features",
                plot_fn=partial(AcousticModelTrainer.plot_world_features,
                                contains_deltas=False,
                                num_coded_sps=hparams.num_coded_sps,
                                num_bap=hparams.num_baps,
                                mgc_label="Predicted spectrogram",
                                lf0_label="Predicted LF0",
                                vuv_label="Predicted VUV",
                                vuv_colour_alpha=('0.8', 1.0)),
                plotter_name=".Org-PyTorch",
                post_processed=True,
            ),
            DataPlotter.Config(
                feature_name="acoustic_features",
                plot_fn=partial(AcousticModelTrainer.plot_world_features,
                                contains_deltas=False,
                                num_coded_sps=hparams.num_coded_sps,
                                num_bap=hparams.num_baps),
                plotter_name=".Org-PyTorch",
                post_processed=True,
                grid_indices=[0, 2]
            )
        ]

    def synth(self,
              hparams: ExtendedHParams,
              ids_input: Union[str, List[str], Tuple[str, ...], os.PathLike],
              post_processing_mapping: Dict[str, str] = None,
              plotter_configs: List[DataPlotter.Config] = None):

        if post_processing_mapping is None:
            post_processing_mapping = {"pred_acoustic_features": "cmp_features",
                                       "acoustic_features": "cmp_features"}

        if plotter_configs is None:
            plotter_configs = AcousticModelTrainer._get_legacy_plotter_configs(
                hparams)

        if not hparams.has_value("synth_feature_names"):
            hparams = copy.deepcopy(hparams)
            hparams.add_hparam("synth_feature_names", ["pred_acoustic_features"])

        return super().synth(hparams=hparams,
                             ids_input=ids_input,
                             post_processing_mapping=post_processing_mapping,
                             plotter_configs=plotter_configs)

    def compute_score(self, data, output, hparams):

        dict_original_post = self.get_output_dict(
            data.keys(), hparams,
            chunk_size=hparams.get_value("n_frames_per_step", default=1))

        metric_dict = {}
        for label_name in next(iter(data.values())).keys():
            metric = Metrics(hparams.metrics)
            for id_name, labels in data.items():
                labels = labels[label_name]
                output = WorldFeatLabelGen.convert_to_world_features(
                    sample=labels,
                    contains_deltas=False,
                    num_coded_sps=hparams.num_coded_sps,
                    num_bap=hparams.num_bap)

                org = WorldFeatLabelGen.convert_to_world_features(
                    sample=dict_original_post[id_name],
                    contains_deltas=hparams.add_deltas,
                    num_coded_sps=hparams.num_coded_sps,
                    num_bap=hparams.num_bap)

                current_metrics = metric.get_metrics(hparams.metrics, *org,
                                                     *output)
                metric.accumulate(id_name, current_metrics)

            metric.log()
            metric_dict[label_name] = metric.get_cum_values()

        return metric_dict

    def get_output_dict(self, id_list, hparams, chunk_size=1):
        assert hparams.has_value("world_dir"), \
            "hparams.world_dir must be set for this operation."
        dict_original_post = dict()
        for id_name in id_list:
            sample = WorldFeatLabelGen.load_sample(
                id_name,
                dir_out=hparams.world_dir,
                add_deltas=hparams.add_deltas,
                num_coded_sps=hparams.num_coded_sps,
                sp_type=hparams.sp_type,
                num_bap=hparams.num_bap,
                load_sp=hparams.load_sp,
                load_lf0=hparams.load_lf0,
                load_vuv=hparams.load_vuv,
                load_bap=hparams.load_bap)
            if chunk_size > 1:
                sample = WorldFeatLabelGen.pad(
                    None, sample, _get_padding_sizes(sample, chunk_size),
                    pad_mode='constant')
            dict_original_post[id_name] = sample
        return dict_original_post

    def synthesize(self, data, hparams, id_list):
        """
        Depending on hparams override the network output with the
        extracted features, then continue with normal synthesis pipeline.
        """
        if hparams.has_value("synth_feature_names"):
            feature_names = hparams.synth_feature_names
            if type(feature_names) not in [list, tuple]:
                feature_names = (feature_names,)
        else:
            feature_names = ("pred_acoustic_features",)
            feature_names = list(next(iter(data.values())).keys())
            self.logger.warning(
                "hparams.synth_feature_names is not defined, using {} instead."
                .format(feature_names))

        self.logger.info("Synthesise from {}".format(", ".join(feature_names)))

        for id_name, features in data.items():
            selected_features = list()
            for feature_name in feature_names:
                selected_features.append(features[feature_name])
            selected_features = np.concatenate(selected_features, axis=1)
            data[id_name] = selected_features

        load_any_org_features = hparams.synth_load_org_sp\
            or hparams.synth_load_org_lf0\
            or hparams.synth_load_org_vuv\
            or hparams.synth_load_org_bap

        if load_any_org_features:
            for id_name, labels in data.items():
                assert hparams.has_value("world_dir"), \
                    "hparams.world_dir must be set for this operation."
                world_dir = hparams.world_dir
                org_labels = WorldFeatLabelGen.load_sample(
                    id_name, world_dir, num_coded_sps=hparams.num_coded_sps,
                    num_bap=hparams.num_bap)

                len_org = len(org_labels)
                len_diff = len_org - len(labels)
                if len_diff > 0:
                    org_labels = WorldFeatLabelGen.trim_end_sample(
                        org_labels, int(len_diff / 2), reverse=True)
                    org_labels = WorldFeatLabelGen.trim_end_sample(
                        org_labels, len_diff - int(len_diff / 2))

                if hparams.synth_load_org_sp:
                    data[id_name][:len_org, :hparams.num_coded_sps] = \
                        org_labels[:, :hparams.num_coded_sps]
                if hparams.synth_load_org_lf0:
                    data[id_name][:len_org, -3] = org_labels[:, -3]

                if hparams.synth_load_org_vuv:
                    data[id_name][:len_org, -2] = org_labels[:, -2]

                if hparams.synth_load_org_bap:
                    if hparams.num_bap == 1:
                        data[id_name][:len_org, -1] = org_labels[:, -1]
                    else:
                        data[id_name][:len_org, -hparams.num_bap:] = \
                            org_labels[:, -hparams.num_bap:]

        super().gen_waveform(id_list=id_list, data=data, hparams=hparams)

    def copy_synth(self, hparams, id_list):

        if not hparams.has_value("synth_feature_names"):
            hparams.setattr_no_type_check("synth_feature_names",
                                          "acoustic_features")

        super().copy_synth(hparams=hparams, id_list=id_list)
