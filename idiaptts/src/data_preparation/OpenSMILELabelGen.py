#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


# System imports.
import arff
import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Tuple

# Third-party imports.
import numpy as np

# Local source tree imports.
from idiaptts.src.data_preparation.LabelGen import LabelGen
from idiaptts.misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor


class OpenSMILELabelGen(LabelGen):
    """Create OpenSMILE features from wav files."""

    @staticmethod
    def gen_data(dir_in: os.PathLike,
                 opensmile_config_file: os.PathLike,
                 feature_name: str,
                 num_frames: int,
                 dir_out: os.PathLike = None,
                 file_id_list: os.PathLike = None,
                 id_list: List[str] = None,
                 file_ext: str = "wav",
                 return_dict: bool = False) -> Tuple:

        if file_id_list is None:
            file_id_list_name = ""
        else:
            id_list, file_id_list_name = OpenSMILELabelGen._get_id_list(
                dir_in, file_id_list, id_list, file_ext)
            if file_id_list_name is not None and file_id_list_name != "":
                file_id_list_name += "-"

        if return_dict:
            label_dict = {}

        normaliser = MeanStdDevExtractor()

        for file_name in id_list:
            features = OpenSMILELabelGen.extract_features(
                config_file=opensmile_config_file,
                file_path=os.path.join(dir_in, file_name + "." + file_ext),
                num_frames=num_frames
            )

            if return_dict:
                label_dict[file_name] = features

            normaliser.add_sample(features)

            if dir_out is not None:
                out_file_path = os.path.join(dir_out, file_name)
                OpenSMILELabelGen._save_to_npz(
                    file_path=out_file_path,
                    features=features.astype(np.float32),
                    feature_name=feature_name)

        if dir_out is not None:
            norm_file_path = os.path.join(dir_out,
                                          file_id_list_name + feature_name)
            logging.info("Write norm_prams to {}".format(norm_file_path))
            normaliser.save(norm_file_path)

        mean, std_dev = normaliser.get_params()
        if return_dict:
            return label_dict, mean, std_dev
        else:
            return mean, std_dev

    @staticmethod
    def _get_id_list(dir_in: os.PathLike, file_id_list: os.PathLike,
                     id_list: List[str] = None, file_ext: str = ".wav"
                     ) -> Tuple[List[str], str]:
        """
        Fill file_id_list by files in dir_in with file_ext if not given and set
        an appropriate file_id_list_name.
        """
        if id_list is None:
            id_list = list()
            filenames = glob.glob(os.path.join(dir_in, "*" + file_ext))
            for filename in filenames:
                id_list.append(os.path.splitext(os.path.basename(filename))[0])
            file_id_list_name = "all"
        else:
            file_id_list_name = os.path.splitext(os.path.basename(file_id_list))[0]

        return id_list, file_id_list_name

    @staticmethod
    def extract_features(config_file: os.PathLike, file_path: os.PathLike,
                         num_frames: int = None) -> np.ndarray:
        """
        Extract features with SMILEExtract.
        Removes first and last generated feature.
        """

        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, "test_output.arff")
        try:
            cmd = ["opensmile/bin/SMILExtract", "-C", config_file,
                   "-I", file_path, "-O", path, "-l", "1"]
            logging.debug(cmd)
            subprocess.check_output(cmd)

            dataset = arff.load(open(path, 'r'))
            data = dataset['data']
        except subprocess.CalledProcessError as e:
            print("SMILExtract stdout output:\n", e.output)
            raise
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if num_frames is None or num_frames == -1:
            features = data
        else:
            len_diff = len(data) - num_frames
            if len_diff > 0:
                features = data[:num_frames]
            else:
                num_features = len(data[0])
                padding = abs(len_diff) * [num_features * [0]]
                features = data + padding

        features = np.atleast_2d(np.asarray(features))[:, 1:-1].astype(float)

        return features


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-a", "--dir_audio",
                        help="Directory containing the audio (wav) files.",
                        type=str, dest="dir_audio", required=True)
    parser.add_argument('-c', '--config_file', default=None,
                        help='Path to the openSMILE config to use.',
                        required=True)
    parser.add_argument("-f", "--num_frames", default=-1,
                        help="The features are cropped/padded to this length.",
                        type=int, dest="num_frames", required=False)
    parser.add_argument("-i", "--file_id_list", default=None,
                        help="Path to text file with ids to process.",
                        type=str, dest="file_id_list", required=False)
    parser.add_argument("--id_name", default=None,
                        help="Single id_name to process",
                        type=str, dest="id_name", required=False)
    parser.add_argument("-n", "--feature_name",
                        help="Name of the feature used to store in npz file.",
                        type=str, required=True)
    parser.add_argument("-o", "--dir_out",
                        help="Output directory to store the labels.",
                        type=str, dest="dir_out", required=True)

    # Parse arguments
    args = parser.parse_args()

    dir_audio = os.path.abspath(args.dir_audio)
    opensmile_config_file = os.path.abspath(args.config_file)
    num_frames = int(args.num_frames)
    feature_name = args.feature_name
    dir_out = os.path.abspath(args.dir_out)

    if args.file_id_list is not None:
        file_id_list = os.path.abspath(args.file_id_list)
        with open(file_id_list) as f:
            id_list = f.readlines()
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]  # Trim entries in-place.
    elif args.id_name is not None:
        file_id_list = None
        id_list = [args.id_name]
    else:
        raise RuntimeError("Either file_id_list or id_name has to be given.")

    assert num_frames == -1 or num_frames > 0, "num_frames has to be positive or -1."
    OpenSMILELabelGen.gen_data(
        dir_in=dir_audio,
        dir_out=dir_out,
        file_id_list=file_id_list,
        id_list=id_list,
        opensmile_config_file=opensmile_config_file,
        feature_name=feature_name,
        num_frames=num_frames,
        return_dict=False
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
