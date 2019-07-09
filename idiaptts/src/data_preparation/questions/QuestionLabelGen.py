#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
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
import numpy as np

# Third-party imports.

# Local source tree imports.
from idiaptts.misc.normalisation.MinMaxExtractor import MinMaxExtractor as NormExtractor
from idiaptts.src.data_preparation.LabelGen import LabelGen
from idiaptts.src.data_preparation.questions.label_normalisation import HTSLabelNormalisation
from idiaptts.misc.utils import makedirs_safe, file_len


class QuestionLabelGen(LabelGen):
    """Create question labels for .lab files."""

    ext_question = ".questions"
    logger = logging.getLogger(__name__)

    def __init__(self, dir_labels, num_questions):

        # Attributes.
        self.dir_labels = dir_labels
        self.num_questions = num_questions
        self.norm_params = None

    def __getitem__(self, id_name):
        """Return the preprocessed sample with the given id_name."""
        sample = self.load_sample(id_name, self.dir_labels, self.num_questions)
        sample = self.preprocess_sample(sample)

        return sample

    @staticmethod
    def trim_end_sample(sample, length, reverse=False):
        """
        Trim the end of a sample by the given length. If reverse is True, the front of the sample is trimmed.
        This function is called after preprocess_sample.
        """
        if length == 0:
            return sample

        if reverse:
            return sample[length:, ...]
        else:
            return sample[:-length, ...]

    def preprocess_sample(self, sample, norm_params=None):
        """
        Normalise one sample (by default to min=0 and max=1). This function should be used within the
        batch loading of PyTorch.

        :param sample:            The sample to pre-process.
        :param norm_params:       Use this normalisation parameters instead of self.norm_params.
        :return:                  Pre-processed sample.
        """

        if norm_params is not None:
            samples_min, samples_max = norm_params
        elif self.norm_params is not None:
            samples_min, samples_max = self.norm_params
        else:
            self.logger.error("Please give norm_params argument or call get_normaliations_params() before.")
            return None

        # Prevent division by zero.
        for index in range(len(samples_min)):
            if samples_min[index] == samples_max[index]:
                samples_max[index] += 1
            elif samples_min[index] > samples_max[index]:
                logging.warning("Min greater then max for feature " + str(index) + ": ("
                                + str(samples_min[index]) + " > " + str(samples_max[index]) + "), changing max to "
                                + str(samples_min[index] + 1))
                samples_max[index] = samples_min[index] + 1

        # Return normalised questions.
        return np.float32((sample - samples_min) / (samples_max - samples_min))

    def postprocess_sample(self, sample, norm_params=None):
        """
        Denormalise one sample. This function is used after inference of a network.

        :param sample:            The sample to post-process.
        :param norm_params:       Use this normalisation parameters instead of self.norm_params.
        :return:                  Post-processed sample.
        """
        if norm_params is not None:
            samples_min, samples_max = norm_params
        elif self.norm_params is not None:
            samples_min, samples_max = self.norm_params
        else:
            self.logger.error("Please give norm_params argument or call get_normaliations_params() before.")
            return None

        # Prevent multiplication by zero.
        for index in range(len(samples_min)):
            if samples_min[index] == samples_max[index]:
                samples_max[index] += 1
            elif samples_min[index] > samples_max[index]:
                logging.warning("Min greater then max for feature " + str(index) + ": ("
                                + str(samples_min[index]) + " > " + str(samples_max[index]) + "), changing max to "
                                + str(samples_min[index] + 1))
                samples_max[index] = samples_min[index] + 1

        sample = np.copy(sample * (samples_max - samples_min) + samples_min)

        return sample

    @staticmethod
    def load_sample(id_name, dir_out, num_questions=425):
        """
        Load labels from dir_out int a numpy array.

        :param id_name:         Id of the sample.
        :param dir_out:         Directory containing the sample.
        :param num_questions:   Required to reshape the array.

        :return:                Numpy array with dimensions num_frames x num_questions.
        """
        id_name = os.path.splitext(os.path.basename(id_name))[0]  # Features should be stored in same directory, no speaker dependent subdirs.
        label_file = os.path.join(dir_out, id_name + QuestionLabelGen.ext_question)

        return np.fromfile(label_file, dtype=np.float32).reshape(-1, num_questions)

    def get_normalisation_params(self, dir_out, file_name=None):
        """
        Read the min max values from a file.
        Saves them in self.norm_params.

        :param dir_out:       Directory containing the normalisation file.
        :param file_name:     Prefix of normalisation file.
                              Expects file to be named <file_name-><self.file_name_appendix>.bin
        :return:              Tuple of normalisation parameters (min, max).
        """

        # Load the normalisation parameters.
        full_file_name = (file_name + "-" if file_name is not None else "") + NormExtractor.file_name_appendix + ".bin"
        self.norm_params = NormExtractor.load(os.path.join(dir_out, full_file_name))

        return self.norm_params

    @staticmethod
    def gen_data(dir_in, file_questions, dir_out=None, file_id_list=None, id_list=None, return_dict=False):
        """
        Generate question labels from HTK labels.

        :param dir_in:         Directory containing the HTK labels.
        :param file_questions: Full file path to the question file.
        :param dir_out:        Directory to store the question labels. If None, labels are not saved.
        :param file_id_list:   Name of the file containing the ids. Normalisation parameters are saved using
                               this name to differentiate parameters between subsets.a
        :param id_list:        The list of utterances to process.
                               Should have the form uttId1 \\n uttId2 \\n ...\\n uttIdN.
                               If None, all file in audio_dir are used.
        :param return_dict:    If true, returns an OrderedDict of all samples as first output.
        :return:               Returns two normalisation parameters as tuple. If return_dict is True it returns
                               all processed labels in an OrderedDict followed by the two normalisation parameters.
        """

        # Fill file_id_list by .lab files in dir_in if not given and set an appropriate file_id_list_name.
        if id_list is None:
            id_list = list()
            filenames = glob.glob(os.path.join(dir_in, "*.lab"))
            for filename in filenames:
                id_list.append(os.path.splitext(os.path.basename(filename))[0])
            file_id_list_name = "all"
        else:
            file_id_list_name = os.path.splitext(os.path.basename(file_id_list))[0]
            id_list = ['{}'.format(os.path.basename(element)) for element in id_list]  # Ignore full path.

        # Create directories in dir_out if it is given.
        if dir_out is not None:
            makedirs_safe(dir_out)

        # Get question generation class.
        label_operater = HTSLabelNormalisation(file_questions)
        if return_dict:
            label_dict, norm_params = label_operater.perform_normalisation(file_id_list_name, id_list, dir_in, dir_out, return_dict=True)
            # self.norm_params = (samples_min, samples_max)
            return label_dict, norm_params[0], norm_params[1]
        else:
            norm_params = label_operater.perform_normalisation(file_id_list_name, id_list, dir_in, dir_out, return_dict=False)
            return norm_params[0], norm_params[1]

    def get_HTK_label_timings_ms(self, htk_label):
        """Print the start and end frame of a label in ms. Unused in current version."""
        # Label example:
        # 118200000 118250000 t^dh-ax+t=ae@2_1/A:1_0_3/B:0-0-2@1-1&2-7#1-4$1-4!1-1;0-1|ax/C:1+1+2/D:in_1/E:det+1@2+3&1+2#0+1/F:content_3/G:10_9/H:8=4@6=3|L-L%/I:6=5/J:72+43-8[2]
        label = htk_label.split()

        return [int(label[0]) / 1000, int(label[1]) / 1000]

    @staticmethod
    def questions_to_phoneme_indices(questions, np_phoneme_indices_in_question):
        """
        Convert questions to indices of phonemes.

        :param questions:                          Questions for one sample.
        :param np_phoneme_indices_in_question:     Numpy array of indices that point to all pure phoneme questions.
        :return:                                   Numpy array with indices of phonemes per frame.
        """
        return questions[:, np_phoneme_indices_in_question].argmax(axis=1)

    @staticmethod
    def questions_to_phoneme_per_frame(questions, np_phoneme_indices_in_question_file, question_file):
        """
        Creates a numpy array with the name of the phoneme for each frame.

        :param questions:                               Questions for one sample.
        :param np_phoneme_indices_in_question_file:     Numpy array of indices that point to all pure phoneme questions.
        :param question_file:                           Full path to the question file used to create the questions.
        :return:                                        Numpy array with phoneme name of each frame.
        """
        fid = open(question_file)

        question_file_list = list()
        for line in np.array(fid.readlines())[np_phoneme_indices_in_question_file]:
            question_file_list.append(line.split()[1].replace('C_', '').replace('C-', '').replace('\"', ''))
        np_question_file = np.array(question_file_list)

        np_indices = QuestionLabelGen.questions_to_phoneme_indices(questions, np_phoneme_indices_in_question_file)

        return np_question_file[np_indices]

    @staticmethod
    def questions_to_phonemes(questions, np_phoneme_indices_in_question_file, question_file):
        """
        Converts a question file to a 2D numpy array with (frame number, phoneme name). The frame number specifies
        the frame from where the phoneme is present. The phoneme ends at the start frame of the next phoneme.

        :param questions:                             Questions for one sample.
        :param np_phoneme_indices_in_question_file:   Numpy array of indices that point to all pure phoneme questions.
        :param question_file:                         Full path to the question file used to create the questions.
        :return:                                      2D numpy array with frame number, phoneme name.
        """
        np_phonemes_per_frame = QuestionLabelGen.questions_to_phoneme_per_frame(questions, np_phoneme_indices_in_question_file, question_file)

        phonemes = list()
        phonemes.append((0, np_phonemes_per_frame[0]))
        for index, phoneme in enumerate(np_phonemes_per_frame):
            if phonemes[-1][1] != phoneme:
                phonemes.append((index, phoneme))
        return np.array(phonemes)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-l", "--dir_labels", help="Directory containing the label (HTK full labels, *.lab) files.",
                        type=str, dest="dir_labels", required=True)
    parser.add_argument("-q", "--file_questions", help="Full path to question file.",
                        type=str, dest="file_questions", required=True)
    # parser.add_argument("-n", "--num_questions", help="Number of questions in the question file.",
    #                     type=int, dest="num_questions", default=425)
    parser.add_argument("-i", "--file_id_list_path",
                        help="Path to text file to read the ids of the files to process.\
                              Default uses all .lab files in the given label_dir.",
                        type=str, dest="file_id_list_path", default=None)
    parser.add_argument("-o", "--dir_out", help="Output directory to store the labels.",
                        type=str, dest="dir_out", required=True)

    # Parse arguments
    args = parser.parse_args()

    dir_labels = os.path.abspath(args.dir_labels)
    file_questions = os.path.abspath(args.file_questions)
    if args.file_id_list_path is not None:
        file_id_list_path = os.path.abspath(args.file_id_list_path)
        # Read which files to process.
        with open(file_id_list_path) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
    else:
        id_list = None

    dir_out = os.path.abspath(args.dir_out)

    # Execute main functionality.
    QuestionLabelGen.gen_data(dir_labels, file_questions, dir_out=dir_out, file_id_list=args.file_id_list_path, id_list=id_list, return_dict=False)

    # DEBUG
    # label_dict, _, _ = QuestionLabelGen.gen_dict(dir_labels, file_questions, dir_out=dir_out,
    #                                        file_id_list=args.file_id_list_path, id_list=id_list, return_dict=True)
    # num_questions = file_len(file_questions) + 9
    # questions_gen = QuestionLabelGen(dir_out, num_questions)
    # # Retrieve an appropriate name for the file_id_list.
    # if file_id_list_path is not None:
    #     file_id_list_name = os.path.splitext(os.path.basename(file_id_list_path))[0]
    # else:
    #     file_id_list_name = "all"
    #
    # # Call this once before starting the preprocessing.
    # norm_params = questions_gen.get_normalisation_params(dir_out, file_name=file_id_list_name)
    #
    # np.set_printoptions(threshold=np.inf)
    # test_label = label_dict["roger_5535"]
    # print(test_label[400, -20:])
    # test_label = questions_gen.preprocess_sample(test_label)
    # print(test_label[400, -20:])
    # test_label = questions_gen.postprocess_sample(test_label)
    # print(test_label[400, -20:])

    sys.exit(0)


if __name__ == "__main__":
    main()
