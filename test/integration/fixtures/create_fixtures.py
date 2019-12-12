#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
Recreate features (except labels) from wav files in database/wav.
Load all models and save them again as checkpoints.
"""

# System imports.
import os
import logging
import shutil
import copy

# Third-party imports.
import numpy as np

# Local source tree imports.
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
from idiaptts.src.data_preparation.phonemes.PhonemeDurationLabelGen import PhonemeDurationLabelGen
from idiaptts.src.data_preparation.wcad.AtomVUVDistPosLabelGen import AtomVUVDistPosLabelGen
from idiaptts.misc.utils import makedirs_safe

logging.basicConfig(level=logging.INFO)
extract_features = False
save_models = True
retrain_models = False

dir_data = os.path.realpath("database")
dir_wav = os.path.join(dir_data, "wav")
file_id_list_path = os.path.join(dir_data, "file_id_list.txt")
with open(file_id_list_path) as f:
    id_list = f.readlines()
id_list = [s.strip(' \t\n\r') for s in id_list]  # Trim line endings in-place.

dir_labels = os.path.join(os.path.realpath("labels"), "label_state_align")
dir_dur = os.path.realpath("dur")
dir_questions = "questions"
dir_world = os.path.realpath("WORLD")
thetas = np.arange(0.03, 0.155, 0.03)
dir_atoms = "wcad-" + "_".join(map("{:.3f}".format, thetas))

if extract_features:
    # Generate labels.
    # # shutil.rmtree(dir_labels)
    # makedirs_safe(dir_labels)
    logging.warning("Label files are not recreated.")
    # TODO: Possible implementation at TTSModel.run_DM_AM().

    # Generate durations
    logging.info("Create duration files.")
    shutil.rmtree(dir_dur)
    makedirs_safe(dir_dur)
    PhonemeDurationLabelGen.gen_data(dir_labels, dir_dur, id_list=id_list)

    # Generate questions.
    logging.info("Create question files.")
    shutil.rmtree(dir_questions)
    makedirs_safe(dir_questions)
    QuestionLabelGen.gen_data(dir_labels, "questions-en-radio_dnn_400.hed", dir_questions, id_list=id_list)

    # Generate WORLD features.
    logging.info("Create WORLD files.")
    shutil.rmtree(dir_world)
    makedirs_safe(dir_world)
    world_generator = WorldFeatLabelGen(dir_world, add_deltas=False, num_coded_sps=20, sp_type="mcep")
    world_generator.gen_data(dir_wav, dir_world, id_list=id_list)
    world_generator = WorldFeatLabelGen(dir_world, add_deltas=True, num_coded_sps=20, sp_type="mcep")
    world_generator.gen_data(dir_wav, dir_world, id_list=id_list)

    # Generate atoms.
    logging.info("Create atom files.")
    shutil.rmtree(dir_atoms)
    makedirs_safe(dir_atoms)
    atom_generator = AtomVUVDistPosLabelGen(os.path.join(os.path.dirname(os.environ["IDIAPTTS_ROOT"]), "tools", "wcad"),
                                            dir_atoms,
                                            dir_world,
                                            thetas)
    atom_generator.gen_data(dir_wav, dir_atoms, id_list=id_list)

if retrain_models:
    raise NotImplementedError("Did not yet implemented retraining of models.")
elif save_models:
    from idiaptts.src.neural_networks.pytorch.ModelHandlerPyTorch import ModelHandlerPyTorch

    from idiaptts.src.model_trainers.wcad.AtomVUVDistPosModelTrainer import AtomVUVDistPosModelTrainer
    hparams = AtomVUVDistPosModelTrainer.create_hparams()
    hparams.model_name = "test_model_in409_out7.nn"
    model_handler = ModelHandlerPyTorch()
    # The following code uses the load_model method and saves it back as a checkpoint.
    # model, model_type, dim_in, dim_out = model_handler.load_model(hparams.model_name, hparams)
    # model_handler.model_type = "RNNDYN-1_RELU_32-1_FC_7"
    # model_handler.dim_in = model.dim_in
    # model_handler.dim_out = model.dim_out
    # model_handler.model_name = hparams.model_name
    # model_handler.model = model
    # model_handler.save_checkpoint(os.path.realpath(hparams.model_name), 3)
    epochs = model_handler.load_checkpoint(hparams.model_name, hparams)
    model_handler.save_checkpoint(os.path.realpath(hparams.model_name), epochs)

    from idiaptts.src.model_trainers.wcad.AtomNeuralFilterModelTrainer import AtomNeuralFilterModelTrainer
    hparams = AtomNeuralFilterModelTrainer.create_hparams()
    hparams.model_name = "neural_filters_model_in409_out2.nn"
    hparams.atom_model_path = "test_model_in409_out7.nn"
    hparams.optimiser_args["lr"] = 0.002
    hparams.thetas = thetas
    hparams.complex_poles = False
    hparams.hparams_atom = copy.deepcopy(hparams)
    model_handler = ModelHandlerPyTorch()
    epochs = model_handler.load_checkpoint(hparams.model_name, hparams)
    model_handler.save_checkpoint(os.path.realpath(hparams.model_name), epochs)

    from idiaptts.src.model_trainers.wcad.PhraseAtomNeuralFilterModelTrainer import PhraseAtomNeuralFilterModelTrainer
    hparams = PhraseAtomNeuralFilterModelTrainer.create_hparams()
    hparams.model_name = "phrase_neural_filters_model_in409_out2.nn"
    hparams.complex_poles = False
    hparams.thetas = thetas
    hparams.hparams_flat = copy.deepcopy(hparams)
    hparams.hparams_atom = copy.deepcopy(hparams)
    hparams.hparams_flat.hparams_atom = hparams.hparams_atom
    hparams.flat_model_path = "neural_filters_model_in409_out2.nn"
    hparams.atom_model_path = "test_model_in409_out7.nn"
    hparams.hparams_atom.model_path = hparams.atom_model_path
    hparams.hparams_flat.model_path = hparams.flat_model_path
    hparams.hparams_flat.atom_model_path = hparams.atom_model_path
    model_handler = ModelHandlerPyTorch()
    epochs = model_handler.load_checkpoint(hparams.model_name, hparams)
    model_handler.save_checkpoint(os.path.realpath(hparams.model_name), epochs)

    from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer
    hparams = AcousticModelTrainer.create_hparams()
    hparams.model_name = "test_model_in409_out67.nn"
    model_handler = ModelHandlerPyTorch()
    epochs = model_handler.load_checkpoint(hparams.model_name, hparams)
    model_handler.save_checkpoint(os.path.realpath(hparams.model_name), epochs)

    from idiaptts.src.model_trainers.WaveNetVocoderTrainer import WaveNetVocoderTrainer
    hparams = WaveNetVocoderTrainer.create_hparams()
    hparams.model_name = "r9y9_wavenet_in23_out128.nn"
    hparams.num_coded_sps = 20
    hparams.input_type = "mulaw-quantize"
    hparams.quantize_channels = 128
    hparams.mu = 127
    hparams.out_channels = hparams.quantize_channels
    hparams.cin_channels = hparams.num_coded_sps + 3
    hparams.upsample_conditional_features = True
    hparams.upsample_scales = [1]
    hparams.layers = 4
    hparams.stacks = 2
    hparams.residual_channels = 2
    hparams.gate_channels = 2
    hparams.skip_out_channels = 2
    hparams.kernel_size = 2
    model_handler = ModelHandlerPyTorch()
    epochs = model_handler.load_checkpoint(hparams.model_name, hparams)
    model_handler.save_checkpoint(os.path.realpath(hparams.model_name), epochs)
