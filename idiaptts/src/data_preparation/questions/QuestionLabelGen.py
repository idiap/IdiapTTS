#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Create question labels from HTK labels.
"""

# System imports.
import argparse
import glob
import logging
import os
import sys
from typing import Callable, Dict, List, Tuple, Union

# Third-party imports.
import numpy as np

# Local source tree imports.
from idiaptts.misc.normalisation.MinMaxExtractor import MinMaxExtractor as NormExtractor
from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.data_preparation.LabelGen import LabelGen
from idiaptts.src.data_preparation.NpzDataReader import NpzDataReader
from idiaptts.src.data_preparation.questions.label_normalisation import HTSLabelNormalisation


class QuestionLabelGen(NpzDataReader, LabelGen):
    """Create question labels for .lab files."""

    ext_question = ".questions"
    logger = logging.getLogger(__name__)

    class Config(NpzDataReader.Config):

        def __init__(self, *args,
                     norm_type: str = NpzDataReader.Config.NormType.MIN_MAX,
                     **kwargs) -> None:

            super().__init__(*args, norm_type=norm_type, **kwargs)

        def create_reader(self):
            reader = QuestionLabelGen(self)

            reader.get_normalisation_params()

            return reader

    def __init__(self, *args, **kwargs):

        if len(args) == 1 and isinstance(args[0], QuestionLabelGen.Config):
            config = args[0]

            super().__init__(config)

            self.legacy_getitem = False
        else:
            # LEGACY support
            if len(args) > 0:
                dir_labels = args[0]
                if len(args) > 1:
                    num_questions = args[1]
                else:
                    num_questions = kwargs["num_questions"]
            else:
                dir_labels = kwargs["dir_labels"]
                num_questions = kwargs["num_questions"]

            super().__init__(
                config=QuestionLabelGen._get_npz_reader_config(dir_labels)
            )
            self.num_questions = num_questions

            self.legacy_getitem = True

        if type(self.directory) in [tuple, list]:
            self.dir_labels = self.directory[0]
        else:
            self.dir_labels = self.directory

    @staticmethod
    def _get_npz_reader_config(dir_labels):
        return NpzDataReader.Config(
            name="questions",
            directory=dir_labels,
            # features="_questions",
            norm_type=NpzDataReader.Config.NormType.MIN_MAX
        )

    def __getitem__(self, id_name):
        """Return the preprocessed sample with the given id_name."""
        sample_dict = super().__getitem__(id_name)

        if self.legacy_getitem:
            # LEGACY support
            return sample_dict["questions"]
        else:
            return sample_dict

    @staticmethod
    def load_sample(id_name, dir_out=None, num_questions=None):
        """
        Load labels from dir_out in a numpy array.

        :param id_name:         Id of the sample.
        :param dir_out:         Directory containing the sample.
        :param num_questions:   Required to reshape the array.

        :return:                Numpy array with dimensions num_frames x
                                num_questions.
        """

        datareader = QuestionLabelGen(dir_out, num_questions)
        return datareader.load(id_name)

    def load(self, id_name: str):
        try:
            return super().load(id_name)
        except FileNotFoundError:
            # LEGACY support
            # Features should be stored in same directory, no speaker
            # dependent subdirectories.
            id_name = os.path.splitext(os.path.basename(id_name))[0]
            label_file = os.path.join(self.directory[0], id_name)
            labels = np.fromfile(label_file + QuestionLabelGen.ext_question,
                                 dtype=np.float32)
            labels = labels.reshape(-1, self.num_questions)
            return labels

    def _load_normalisation_params(self, directory, file_name):
        try:
            return super()._load_normalisation_params(directory,
                                                      file_name=file_name)
        except FileNotFoundError:
            # LEGACY support
            if directory is None:
                directory = self.dir_labels

            if file_name is None:
                file_name = ""
            elif os.path.basename(file_name) != "":
                file_name += "-"
            full_file_name = file_name + self.normaliser.file_name_appendix

            self.norm_params = self.normaliser.load(os.path.join(
                directory, full_file_name + ".bin"))

    @staticmethod
    def gen_data(dir_in, file_questions, dir_out=None, file_id_list="",
                 id_list=None, return_dict=False):
        """
        Generate question labels from HTK labels.

        :param dir_in:         Directory containing the HTK labels.
        :param file_questions: Full file path to the question file.
        :param dir_out:        Directory to store the question labels.
                               If None, labels are not saved.
        :param file_id_list:   Name of the file containing the ids.
                               Normalisation parameters are saved using
                               this name to differentiate parameters
                               between subsets.
        :param id_list:        The list of utterances to process. Should
                               have the form uttId1 \\n uttId2 \\n ...
                               \\n uttIdN. If None, all file in
                               audio_dir are used.
        :param return_dict:    If true, returns an OrderedDict of all
                               samples as first output.
        :return:               Returns two normalisation parameters as
                               tuple. If return_dict is True it returns
                               all processed labels in an OrderedDict
                               followed by the two normalisation
                               parameters.
        """

        # Fill file_id_list by .lab files in dir_in, if not given, and
        # set an appropriate file_id_list_name.
        if id_list is None:
            id_list = list()
            filenames = glob.glob(os.path.join(dir_in, "*.lab"))
            for filename in filenames:
                id_list.append(os.path.splitext(os.path.basename(filename))[0])
            file_id_list_name = "all"
        else:
            file_id_list_name = os.path.splitext(
                os.path.basename(file_id_list))[0]
            id_list = ['{}'.format(os.path.basename(element))  # Ignore full path.
                       for element in id_list]

        if dir_out is not None:
            makedirs_safe(dir_out)

        label_operator = HTSLabelNormalisation(file_questions)
        if return_dict:
            label_dict, norm_params = label_operator.perform_normalisation(
                file_id_list_name, id_list, dir_in, dir_out, return_dict=True)
            return label_dict, norm_params[0], norm_params[1]
        else:
            norm_params = label_operator.perform_normalisation(
                file_id_list_name, id_list, dir_in, dir_out, return_dict=False)
            return norm_params[0], norm_params[1]

    def get_HTK_label_timings_ms(self, htk_label):
        """
        Print the start and end frame of a label in ms.
        Unused in current version.
        """
        # Label example:
        # 118200000 118250000 t^dh-ax+t=ae@2_1/A:1_0_3/B:0-0-2@1-1&2-7#1-4$1-4!1-1;0-1|ax/C:1+1+2/D:in_1/E:det+1@2+3&1+2#0+1/F:content_3/G:10_9/H:8=4@6=3|L-L%/I:6=5/J:72+43-8[2]
        label = htk_label.split()

        return [int(label[0]) / 1000, int(label[1]) / 1000]

    @staticmethod
    def questions_to_phoneme_indices(questions, np_phoneme_indices_in_question,
                                     default_index=-1):
        """
        Convert questions to indices of phonemes.

        :param questions:       Questions for one sample.
        :param np_phoneme_indices_in_question:  Numpy array of indices
                                that point to all pure phoneme questions.
        :param default_index:   When no phoneme is present, use the this
                                index instead. This should only happen
                                at the beginning and end of the sample.
        :return:                Numpy array with indices of phonemes per
                                frame.
        """

        max_values = questions[:, np_phoneme_indices_in_question].max(axis=1)
        indices = questions[:, np_phoneme_indices_in_question].argmax(axis=1)
        no_phoneme_flag = max_values == 0
        if no_phoneme_flag.sum() > 0:
            logging.warning("Using default phoneme index {} for frames without"
                            " phoneme information which are {}".format(
                                default_index,
                                np.arange(len(no_phoneme_flag))[no_phoneme_flag]))
        indices[no_phoneme_flag] = default_index
        return indices

    @staticmethod
    def questions_to_phoneme_per_frame(questions,
                                       np_phoneme_indices_in_question_file,
                                       question_file):
        """
        Creates a numpy array with the name of the phoneme for each
        frame.

        :param questions:       Questions for one sample.
        :param np_phoneme_indices_in_question_file: Numpy array of
                                indices that point to all pure phoneme
                                questions.
        :param question_file:   Full path to the question file used to
                                create the questions.
        :return:                Numpy array with phoneme name of each
                                frame.
        """
        fid = open(question_file)

        question_file_list = list()
        all_lines = np.array(fid.readlines())
        phoneme_lines = all_lines[np_phoneme_indices_in_question_file]
        for line in phoneme_lines:
            question_file_list.append(line.split()[1].replace('C_', '')
                                      .replace('C-', '').replace('\"', ''))
        np_question_file = np.array(question_file_list)

        np_indices = QuestionLabelGen.questions_to_phoneme_indices(
            questions, np_phoneme_indices_in_question_file)

        np_question_file = np.insert(np_question_file, 0, '?', axis=0)
        np_indices += 1

        return np_question_file[np_indices]

    @staticmethod
    def questions_to_phonemes(questions, np_phoneme_indices_in_question_file,
                              question_file):
        """
        Converts a question file to a 2D numpy array with (frame number,
        phoneme name). The frame number specifies the frame from where
        the phoneme is present. The phoneme ends at the start frame of
        the next phoneme.

        :param questions:      Questions for one sample.
        :param np_phoneme_indices_in_question_file:  Numpy array of
                               indices that point to all pure phoneme
                               questions.
        :param question_file:  Full path to the question file used to
                               create the questions.
        :return:               2D numpy array with frame number, phoneme
                               name.
        """
        np_phonemes_per_frame = QuestionLabelGen.questions_to_phoneme_per_frame(
            questions, np_phoneme_indices_in_question_file, question_file)

        phonemes = list()
        phonemes.append((0, np_phonemes_per_frame[0]))
        for index, phoneme in enumerate(np_phonemes_per_frame):
            if phonemes[-1][1] != phoneme:
                phonemes.append((index, phoneme))
        return np.array(phonemes)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-l", "--dir_labels",
                        help="Directory containing the label (HTK full labels,"
                        " *.lab) files.", type=str, dest="dir_labels",
                        required=True)
    parser.add_argument("-q", "--file_questions",
                        help="Full path to question file.",
                        type=str, dest="file_questions", required=True)
    parser.add_argument("-i", "--file_id_list_path",
                        help="Path to text file to read the ids of the files "
                        "to process. Default uses all .lab files in the given "
                        "label_dir.", type=str, dest="file_id_list_path",
                        default=None)
    parser.add_argument("-o", "--dir_out",
                        help="Output directory to store the labels.", type=str,
                        dest="dir_out", required=True)

    args = parser.parse_args()

    dir_labels = os.path.abspath(args.dir_labels)
    file_questions = os.path.abspath(args.file_questions)
    if args.file_id_list_path is not None:
        file_id_list_path = os.path.abspath(args.file_id_list_path)

        with open(file_id_list_path) as f:
            id_list = f.readlines()
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
    else:
        id_list = None

    dir_out = os.path.abspath(args.dir_out)

    QuestionLabelGen.gen_data(dir_labels, file_questions, dir_out=dir_out,
                              file_id_list=args.file_id_list_path,
                              id_list=id_list, return_dict=False)

    sys.exit(0)


if __name__ == "__main__":
    main()
