#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Create atom labels for wav files.
"""

# System imports.
import argparse
from collections import OrderedDict
import contextlib  # Create context for audio load.
import glob
import logging
import math
import os
import sys
import time

# Third-party imports
import numpy as np
from scipy import signal
import wave

# Local source tree imports.
from idiaptts.misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor
from idiaptts.misc.normalisation.MinMaxExtractor import MinMaxExtractor
from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.data_preparation.LabelGen import LabelGen


class AtomLabelGen(LabelGen):
    """
    Create wcad atom labels for .wav files.
    Format is |frames| x |thetas| x 2 (amp and theta).
    """
    ext_atoms = ".atoms"
    ext_phrase = ".phrase"
    atom_dim = 2

    logger = logging.getLogger(__name__)

    def __init__(self, wcad_root, dir_labels, thetas, k=6, frame_size_ms=5):
        """
        Constructor to use the class as a dataset and data generator.

        :param wcad_root:      Main directory of wcad, containing wcad.py.
                               This directory is added to sys.path.
        :param dir_labels:     While using it as a database dir_labels
                               has to contain the prepared labels.
        :param k:              K value of the atoms.
        :param thetas:         List of theta values of the atoms.
        :param frame_size_ms:  Length of each frame in ms. Num_frames=
                               total_audio_duration/frame_size_ms, both
                               in ms.
        """

        self.k = k
        self.theta_interval = thetas
        self.num_thetas = len(self.theta_interval)

        self.wcad_root = wcad_root
        self.dir_labels = dir_labels
        self.frame_size_ms = frame_size_ms

        self.norm_params = None

        # Enable wcad imports.
        if not any(wcad_root in p for p in sys.path):
            assert os.path.isfile(os.path.join(os.path.abspath(wcad_root),
                                               "wcad.py")),\
                "Cannot find wcad.py at {}; is the correct WCAD root " \
                "directory given?".format(wcad_root)
            sys.path.append(wcad_root)

    def __getitem__(self, id_name):
        """Return the preprocessed sample with the given id_name."""
        sample = self.load_sample(id_name, self.dir_labels,
                                  len(self.theta_interval))
        sample = self.preprocess_sample(sample)

        return sample

    @staticmethod
    def trim_end_sample(sample, length, reverse=False):
        """
        Trim the end of a sample by the given length. If reverse is
        True, the front of the sample is trimmed. This function is
        called after preprocess_sample. Atoms in the trimmed area are
        moved to the next valid frame. The amplitudes of atoms are
        added, if necessary. The function accepts 2D samples with
        T x |thetas| x 1 (amp) and 3D samples with
        T x |thetas| x 2 (amp, theta).
        """
        if length == 0:
            return sample

        if sample.ndim == 2:
            tmp_amps = np.zeros((sample.shape[1]), dtype=np.float32)

            if reverse:
                for i in range(0, length):
                    # Atom_labels is numpy array with dim
                    # num_frames x num_thetas x 2 (amp & theta).
                    tmp_amps += sample[i, :]
                sample = sample[length:, :]
                # Apply value of sliced atoms to the first label.
                sample[0, :] += tmp_amps
            else:
                for i in range(-length, -1):
                    tmp_amps += sample[i, :]
                sample = sample[:-length, :]
                # Apply value of sliced atoms to the last label.
                sample[-1, :] += tmp_amps

        elif sample.ndim == 3:
            tmp_amps = np.zeros((sample.shape[1]), dtype=np.float32)
            tmp_thetas = np.zeros((sample.shape[1]), dtype=np.float32)

            if reverse:
                for i in range(0, length):
                    # Atom_labels is numpy array with dim
                    # num_frames x num_thetas x 2 (amp & theta).
                    tmp_amps += sample[i, :, 0]
                    tmp_thetas = np.maximum(tmp_thetas, sample[i, :, 1])
                sample = sample[length:, ...]

                # Apply value of sliced atoms to the first label.
                sample[0, :, 0] += tmp_amps
                sample[0, :, 1] = np.maximum(tmp_thetas, sample[0, :, 1])
            else:
                for i in range(-length, -1):
                    tmp_amps += sample[i, :, 0]
                    tmp_thetas = np.maximum(tmp_thetas, sample[i, :, 1])
                sample = sample[:-length, :]
                # Apply value of sliced atoms to the last label.
                sample[-1, :, 0] += tmp_amps
                sample[-1, :, 1] = np.maximum(tmp_thetas, sample[-1, :, 1])

        return sample

    def preprocess_sample(self, sample, norm_params=None):
        """
        Normalise one sample (by default to variance 1). This function
        should be used within the batch loading of PyTorch.

        :param sample:       The sample to pre-process.
        :param norm_params:  Use this normalisation parameters instead
                             of self.norm_params. Default distribution
                             of norm_params is 0 amps, 1 thetas.
        :return:             Pre-processed sample.
        """
        if norm_params is not None:
            # When norm_params are given use the possibly non-zero mean.
            mean, std_dev = norm_params
        elif self.norm_params is not None:
            mean, std_dev = self.norm_params
        else:
            self.logger.error("Please give norm_params argument or call"
                              " get_normaliations_params() before.")
            return None

        # Only take the amplitudes, theta is implicit.
        return np.float32((sample - mean) / std_dev)[:, :, 0]

    def postprocess_sample(self, sample, norm_params=None, identify_peaks=True,
                           peak_range=100):
        """
        Denormalise one sample. This function is used after inference of
        a network.

        :param sample:          The sample to post-process.
        :param norm_params:     Use this normalisation parameters
                                instead of self.norm_params.
        :param identify_peaks:  If true only peaks of sample are returned.
        :param peak_range:      Average width of spike distributions.
        :return:                Post-processed sample.
        """

        if identify_peaks:
            sample = AtomLabelGen.identify_peaks(sample, peak_range)

        if norm_params is not None:
            mean, std_dev = norm_params
        elif self.norm_params is not None:
            mean = self.norm_params[0][0],
            std_dev = self.norm_params[1][0]
        else:
            self.logger.error("Please give norm_params argument or call"
                              " get_normaliations_params() before.")
            return None

        sample = np.copy((sample * std_dev) + mean)

        # amp_peaks = self.identify_peaks(sample)
        # # Only keep maximum values within a certain range.
        # amps_all = amp_peaks.sum(axis=1).reshape(-1, 1)
        # abs_max = maximum_filter1d(np.abs(amps_all), size=30, axis=0)
        # amps_all[np.not_equal(np.abs(amps_all), abs_max)] = 0
        # amp_peaks[amps_all.squeeze() == 0, :] = 0

        thetas = np.tile(np.copy(self.theta_interval), (len(sample), 1))
        thetas[sample == 0] = 0
        if sample.ndim == 1:
            sample = np.concatenate((np.expand_dims(sample, axis=1), thetas),
                                    axis=1)
        else:
            sample = np.stack((sample, thetas), axis=2)

        return sample

    def get_normalisation_params(self, dir_out, file_name=None):
        """
        Read mean std_dev values from a file. Save them in
        self.norm_params.

        :param dir_out:    Directory containing the normalisation file.
        :param file_name:  Prefix of normalisation file. Expects file to
                           be named <file_name-><MeanStdDevExtractor.file_name_appendix>.npz
        :return:           Tuple of normalisation parameters (mean, std_dev).
        """
        if file_name is None:
            file_name = ""
        elif os.path.basename(file_name) != "":
            file_name += "-"

        full_file_name = file_name + MeanStdDevExtractor.file_name_appendix
        try:
            self.norm_params = MeanStdDevExtractor.load(os.path.join(
                dir_out, full_file_name + ".npz"))
        except FileNotFoundError:
            # LEGACY support
            self.norm_params = MeanStdDevExtractor.load(os.path.join(
                dir_out, full_file_name + ".bin"))

        return self.norm_params

    @staticmethod
    def identify_peaks(label, peak_range=60):
        """
        A function to identify the peaks of distributions around each
        predicted spike.

        :param label:       Predicted spike distribution.
        :param peak_range:  Average width of spike distributions.
        :return:            Spiky version of the signal.
        """
        label = np.copy(label)

        for theta_idx in range(label.shape[1]):
            frame_amps = label[:, theta_idx]

            # Skip over those with all zero or all but one zero elements.
            if np.count_nonzero(frame_amps) <= 1:
                continue

            peaks = signal.find_peaks_cwt(frame_amps.squeeze(),
                                          np.arange(1, peak_range))
            mask = np.ones(frame_amps.shape, dtype=bool)
            if len(peaks) > 0:
                mask[peaks] = False
            peaks = signal.find_peaks_cwt(np.negative(frame_amps.squeeze()),
                                          np.arange(1, peak_range))
            if len(peaks) > 0:
                mask[peaks] = False
            frame_amps[mask] = 0

            label[:, theta_idx] = frame_amps

        amps_peaks = np.max(label, axis=1)
        amps_min = np.min(label, axis=1)

        # Find peaks with maximum absolute value.
        mask = amps_peaks < -amps_min
        amps_peaks[mask] = amps_min[mask]

        # Set all non peak elements to zero.
        mask = np.not_equal(
            label,
            np.repeat(amps_peaks.reshape(-1, 1), label.shape[1], axis=1))
        label[mask] = 0

        return label

    @staticmethod
    def index_to_theta(idx, theta_start, theta_step):
        """Convert index in output to corresponding theta value."""
        return idx * theta_step + theta_start

    @staticmethod
    def theta_to_index(theta, thetas):
        """Convert a theta to corresponding index in output."""
        if not isinstance(thetas, np.ndarray):
            thetas = np.array(thetas)

        if theta > 0:
            theta = np.abs(thetas - theta).argmin()
        return theta

    @staticmethod
    def load_sample(id_name, dir_out, num_thetas=9):
        """
        Load labels from dir_out/id_name.npz into numpy array of dim
        T x |thetas| x 2 (amp, theta)."""
        id_name = os.path.splitext(os.path.basename(id_name))[0]
        file_path = os.path.join(dir_out, id_name)
        logging.debug("Load atom features for " + id_name)

        try:
            archive = np.load(file_path + ".npz")
            atoms = archive["atoms"]
        except FileNotFoundError:
            file_path += AtomLabelGen.ext_atoms
            with open(file_path, 'rb') as f:
                atoms = np.fromfile(f, dtype=np.float32)
                atoms = np.reshape(atoms, [-1, num_thetas, 2])

        return atoms

    @staticmethod
    def labels_to_atoms(np_labels, k, frame_size=5, amp_threshold=0.3):
        """
        Transform labels to GammaAtoms. This function expects the spiky
        version of the prediction. See the identify_peaks function.

        :param np_labels:      Numpy array of spiky atom predictions.
        :param k:              K value of the atoms.
        :param frame_size:     Frame size in ms.
        :param amp_threshold:  Minimum amplitude of a spike to be taken
                               as an atom.
        :return:               List of GammaAtoms.
        """
        from tools.wcad.wcad.object_types.atom import GammaAtom
        atoms = list()

        if np_labels.ndim > 2:
            # If multiple thetas exist, iterate them.
            for idx, np_amp_theta in enumerate(np_labels):
                for amp, theta in np_amp_theta:
                    if abs(amp) >= amp_threshold:
                        theta = max(0.005, theta)
                        atoms.append(GammaAtom(k, theta,
                                               int(1000 / frame_size), amp,
                                               idx))
        else:
            # Otherwise process only the one theta value.
            for idx, (amp, theta) in enumerate(np_labels):
                if abs(amp) >= amp_threshold:
                    theta = max(0.005, theta)
                    atoms.append(GammaAtom(k, theta,
                                           int(1000 / frame_size), amp, idx))

        return atoms

    @staticmethod
    def atoms_to_lf0(atoms, num_frames):
        """Generate lf0 from atoms."""
        reconstruction = np.zeros(num_frames)
        for atom in atoms:
            padded_curve = atom.get_padded_curve(num_frames)
            reconstruction += padded_curve

        return reconstruction

    @staticmethod
    def labels_to_lf0(labels, k, frame_size=5, amp_threshold=0.3):
        """
        Generate lf0 from labels by calling atoms_to_lf0(labels_to_atoms(...)).
        """
        return AtomLabelGen.atoms_to_lf0(
            AtomLabelGen.labels_to_atoms(labels, k, frame_size, amp_threshold),
            len(labels))

    @staticmethod
    def get_audio_length(id_name, audio_dir, frame_size_ms):
        """
        Get the number of frames of an audio file with the given frame_size_ms.
        """
        file_path = os.path.join(audio_dir, id_name + ".wav")
        with contextlib.closing(wave.open(file_path)) as f:
            frames = f.getnframes()
            srate = f.getframerate()
            duration = (frames / float(srate)) * 1e3
            duration = int(math.ceil(duration / float(frame_size_ms))) \
                * frame_size_ms
        return duration

    @staticmethod
    def atoms_to_labels(atom_list, thetas, num_frames, label_dimension=2):
        """Convert a list of atoms to labels."""
        np_atom_labels = np.zeros((num_frames, len(thetas), label_dimension),
                                  dtype=np.float32)

        # Apply atoms at their respective position.
        for atom in atom_list:
            np_atom_labels[atom.position, AtomLabelGen.theta_to_index(
                atom.theta, thetas)] += [atom.amp, atom.theta]

        return np_atom_labels

    def gen_data(self, dir_in, dir_out=None, file_id_list="", id_list=None,
                 return_dict=False):
        """
        Prepare atom labels from wav files.
        If id_list is not None, only the ids listed there are generated,
        otherwise for each .wav file in the dir_in. Atoms are computed
        by the wcad algorithm. Examples with more than 70 atoms are
        rejected. One can create a new file_id_list by uncommenting the
        lines before the return statement. Nevertheless, the current
        file_id_list is not substituted by it. The algorithm also saves
        the extracted phrase component in dir_out/id_name.phrase, if
        dir_out is not None.

        :param dir_in:        Directory containing the org wav files.
        :param dir_out:       Directory where the labels are stored. If
                              None, no labels are stored.
        :param file_id_list:  Name of the file containing the ids.
                              Normalisation parameters are saved using
                              this name to differentiate parameters
                              between subsets.
        :param id_list:       The list of utterances to process. Should
                              have the form uttId1 \\n uttId2 \\n ...\\n
                              uttIdN. If None, all wav files in
                              audio_dir are used.
        :param return_dict:   If True, returns an OrderedDict of all
                              samples as first output.
        :return:              Returns mean=0.0, std_dev, min, max of atoms.
        """

        # Fill file_id_list by .wav files in dir_in if not given and set
        # an appropriate file_id_list_name.
        if id_list is None:
            id_list = list()
            filenames = glob.glob(os.path.join(dir_in, "*.wav"))
            for filename in filenames:
                id_list.append(os.path.splitext(os.path.basename(filename))[0])
            file_id_list_name = "all"
        else:
            file_id_list_name = os.path.splitext(os.path.basename(
                file_id_list))[0]

        if dir_out is not None:
            makedirs_safe(dir_out)

        if return_dict:
            label_dict = OrderedDict()

        mean_std_ext_atom = MeanStdDevExtractor()
        min_max_ext_atom = MinMaxExtractor()
        # mean_std_ext_phrase = MeanStdDevExtractor()
        # min_max_ext_phrase = MinMaxExtractor()

        from wcad import WaveInput, PitchExtractor, MultiphraseExtractor, \
            DictionaryGenerator, AtomExtrator, ModelCreator, ModelSaver, \
            Params, Paths

        correct_utts = list()
        self.logger.info("Create atom labels for " + "[{0}]".format(
            ", ".join(str(i) for i in id_list)))
        for id_name in id_list:
            self.logger.debug("Create atom labels for " + id_name)

            # Wcad has to be called in its root directory, therefore a
            # change dir operation is necessary.
            cwd = os.getcwd()
            os.chdir(self.wcad_root)
            args = [os.path.join(dir_in, id_name + ".wav"), dir_out]
            print(args)
            params = Params()
            # Overwrite the possible theta values by selected values.
            params.local_atoms_thetas = self.theta_interval
            params.k = [self.k]
            # params.min_atom_amp = 0.1
            paths = Paths(args, params)

            # Start the extraction process.
            start_t = time.time()
            waveform = WaveInput(paths.wav, params).read()
            pitch = PitchExtractor(waveform, params, paths).compute()

            # Compute the phrase component.
            phrase = MultiphraseExtractor(pitch, waveform, params, paths)\
                .compute()
            phrase_curve = phrase.curve

            # Extract atoms.
            dictionary = DictionaryGenerator(params, paths).compute()
            atoms = AtomExtrator(waveform, pitch, phrase, dictionary,
                                 params, paths).compute()

            # Create a model.
            model = ModelCreator(phrase, atoms, pitch).compute()
            print(('Model created in %s seconds' % (time.time() - start_t)))

            # Save the atoms.
            ModelSaver(model, params, paths).save()
            os.chdir(cwd)

            # Check if output can be correct.
            possible_extraction_failure = False
            if len(atoms) < 50 and not any(a.amp > 10 for a in atoms):
                correct_utts.append(id_name)
            else:
                self.logger.warning("Possible fail of atom extractor for {} "
                                    "(atoms: {}, frames: {}, max: {}).".format(
                                        id_name, len(atoms), len(phrase_curve),
                                        max(a.amp for a in atoms)))
                possible_extraction_failure = True

            atoms.sort(key=lambda x: x.position)

            duration = self.get_audio_length(id_name, dir_in,
                                             self.frame_size_ms)

            # The algorithm generates a few atoms at negative positions,
            # pad them into the first atom at positive position.
            padded_amp = 0
            padded_theta = 0
            for idx, atom in enumerate(atoms):
                if atom.position < 0:
                    padded_amp += atom.amp
                    padded_theta += atom.theta
                else:
                    atoms[idx].amp += padded_amp
                    atoms[idx].theta = (atoms[idx].theta + padded_theta) \
                        / (idx + 1)
                    del atoms[:idx]  # Remove the negative atoms from the list.
                    break

            # The algorithm might also generate a few atoms beyond the
            # last label, pad them into the last label.
            padded_amp = 0
            padded_theta = 0
            for idx, atom in reversed(list(enumerate(atoms))):
                if atom.position * self.frame_size_ms > duration:
                    padded_amp += atom.amp
                    padded_theta += atom.theta
                else:
                    atoms[idx].amp += padded_amp
                    atoms[idx].theta = (atoms[idx].theta + padded_theta) \
                        / (len(atoms) - idx)
                    atoms = atoms[:-(len(atoms) - idx - 1) or None]
                    break

            # Create a label for each frame (size of frame_size_ms) with
            # amplitude and theta of contained atoms.
            np_atom_labels = AtomLabelGen.atoms_to_labels(
                atoms, self.theta_interval, int(duration / self.frame_size_ms))

            np_atom_amps = np.sum(np_atom_labels, axis=1)

            # Only add successful extractions to norm params computation.
            if not possible_extraction_failure:
                # Only compute std_dev from atoms.
                non_zero_atom = np_atom_amps[np_atom_amps[:, 0] != 0.0]
                mean_std_ext_atom.add_sample(non_zero_atom)
                min_max_ext_atom.add_sample(np_atom_amps)
                # mean_std_ext_phrase.add_sample(phrase_curve)
                # min_max_ext_phrase.add_sample(phrase_curve)

            if return_dict:
                label_dict[id_name] = np_atom_labels
            if dir_out is not None:
                # Save phrase, because it might be used in synthesis.
                self._save_to_npz(os.path.join(dir_out, id_name),
                                  phrase_curve.astype(np.float32), "phrase")

                self._save_to_npz(os.path.join(dir_out, id_name),
                                  np_atom_labels.astype(np.float32), "atoms")

        # Manually set mean of atoms to 0, otherwise frames without atom
        # will have an amplitude.
        if mean_std_ext_atom.sum_length > 0:  # Was at least one atom added?
            mean_std_ext_atom.sum_frames[:] = 0.0
        else:
            mean_std_ext_atom.sum_frames = np.zeros(np_atom_amps.shape[1:])
            mean_std_ext_atom.sum_squared_frames = np.zeros(
                np_atom_amps.shape[1:])
        mean_std_ext_atom.sum_squared_frames[1] = mean_std_ext_atom.sum_length\
            * self.theta_interval[-1]

        mean_std_ext_atom.save(os.path.join(dir_out, file_id_list_name))
        min_max_ext_atom.save(os.path.join(dir_out, file_id_list_name))
        # mean_std_ext_phrase.save(os.path.join(dir_out,
        #                                       file_id_list_name + '-phrase'))
        # min_max_ext_phrase.save(os.path.join(dir_out,
        #                                      file_id_list_name + '-phrase'))

        mean_atoms, std_atoms = mean_std_ext_atom.get_params()
        min_atoms, max_atoms = min_max_ext_atom.get_params()
        # mean_phrase, std_phrase = mean_std_ext_phrase.get_params()
        # min_phrase, max_phrase = min_max_ext_atom.get_params()

        # Use this block to save the part of the file_id_list for which
        # atom extraction was successful into a new file.
        if correct_utts:
            corrected_file_id_list_path = os.path.join(
                os.path.dirname(dir_in),
                "wcad_{}.txt".format(os.path.basename(file_id_list_name)))
            with open(corrected_file_id_list_path, 'w') as f:
                f.write('\n'.join(correct_utts) + '\n')

        if return_dict:
            # Return dict of labels for all utterances.
            return label_dict, mean_atoms, std_atoms,  min_atoms, max_atoms
                   # mean_phrase, std_phrase, \
                   # min_phrase, max_phrase
        else:
            return mean_atoms, std_atoms, min_atoms, max_atoms
                   # mean_phrase, std_phrase, \
                   # min_phrase, max_phrase


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-w", "--wcad_root",
                        help="Directory with the WCAD scripts.", type=str,
                        dest="wcad_root", required=True)
    parser.add_argument("-a", "--dir_audio",
                        help="Directory containing the audio (wav) files.",
                        type=str, dest="dir_audio", required=True)
    parser.add_argument("-f", "--frame_size_ms",
                        help="Frame size [ms] of the labels.", type=int,
                        dest="frame_size_ms", default=5)
    parser.add_argument("-i", "--file_id_list_path",
                        help="Path to a text file to read the ids of the files"
                        " to process. Default uses all .wav files in the given"
                        " dir_audio.", type=str, dest="file_id_list_path",
                        default=None)
    parser.add_argument("-o", "--dir_out",
                        help="Output directory to store the labels.", type=str,
                        dest="dir_out", required=True)
    parser.add_argument("-k", help="K value of atoms.", type=int, dest="k",
                        default=6)
    parser.add_argument("--thetas",
                        help="A list of theta values. --theta_{start, stop, "
                        "step} are ignored when given.", nargs='*', type=float,
                        dest="thetas", default=None)
    parser.add_argument("--theta_start", help="Start value of theta.",
                        type=float, dest="theta_start", default=0.01)
    parser.add_argument("--theta_stop", help="Stop value of theta (excluded).",
                        type=float, dest="theta_stop", default=0.055)
    parser.add_argument("--theta_step", help="Distance between the thetas.",
                        type=float, dest="theta_step", default=0.005)

    args = parser.parse_args()

    wcad_root = os.path.abspath(args.wcad_root)
    assert os.path.isfile(os.path.join(wcad_root, "wcad.py")),\
        "Cannot find wcad.py at {}; is the correct WCAD root directory given?"\
        .format(wcad_root)

    dir_audio = os.path.abspath(args.dir_audio)
    frame_size_ms = args.frame_size_ms
    if frame_size_ms != parser.get_default("frame_size_ms"):
        logging.warning("Frame size must match WCAD configuration in "
                        " wcad/object_types/params.py.")

    dir_out = os.path.abspath(args.dir_out)

    k = args.k
    theta_start = args.theta_start
    theta_step = args.theta_step
    theta_stop = args.theta_stop
    if args.thetas is None:
        thetas = np.arange(theta_start, theta_stop, theta_step)
    else:
        thetas = args.thetas

    if args.file_id_list_path is not None:
        file_id_list_path = os.path.abspath(args.file_id_list_path)

        with open(file_id_list_path) as f:
            id_list = f.readlines()
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
    else:
        id_list = None

    atom_gen = AtomLabelGen(wcad_root, dir_out, thetas, k, frame_size_ms)
    atom_gen.gen_data(dir_audio, dir_out, args.file_id_list_path, id_list,
                      return_dict=False)

    sys.exit(0)


if __name__ == "__main__":
    main()
