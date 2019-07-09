#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In-depth Description:
# An MFCC generator which can be called concurrently.
# Computation is done within a temporary directory.
# The default configuration matches with the config used in
# force_align.py. Changes to config have to be reflected in both files.

# Scripts taken from Merlin and adapted.
# https://github.com/CSTR-Edinburgh/merlin

__doc__="""
Generate MFCCs for all ids listed in the given file_id_list. For each id a wav file
has to exist at dir_wav/id.wav. The computed MFCCs are stored in the given dir_mfcc
folder as id.mfc.\n

Examples:\\n

    >>> python3 gen_mfcc.py --dir_wav ./wav/ --dir_mfcc ./mfcc/ --file_id_list ../data/file_id_list.txt

Default configuration is:

%s
"""

DEFAULT_CONFIG = """SOURCEKIND = WAVEFORM
SOURCEFORMAT = WAVE
TARGETRATE = 50000.0
TARGETKIND = MFCC_D_A_0
WINDOWSIZE = 250000.0
PREEMCOEF = 0.97
USEHAMMING = T
ENORMALIZE = T
CEPLIFTER = 22
NUMCHANS = 20
NUMCEPS = 12
"""
__doc__ %= DEFAULT_CONFIG

# System imports.
import os, sys
import logging
import argparse
from subprocess import check_call
import tempfile

# Third-party imports.

# Local source tree imports.
HTKDIR = "../../../tools/bin/htk/"
HCopy  = os.path.join(HTKDIR, 'HCopy' )


class GenMFCC(object):
    """Class description.
    """
    logger = logging.getLogger(__name__)

    # Constants.

    ########################
    # Default constructor
    #
    def __init__(self, dir_wav, dir_mfcc, id_list):
        """Default constructor description.
        """
        self.dir_wav = dir_wav
        self.dir_mfcc = dir_mfcc
        self.id_list = id_list
        
        # Check if mfcc directory exists.
        if not os.path.exists(self.dir_mfcc):
            os.makedirs(self.dir_mfcc)

    def _HCopy(self, config_file):
        """
        Compute MFCCs
        """
        print("Generate MFCCs.")
        with tempfile.TemporaryDirectory() as dir_tmp:
            self.dir_cfg = os.path.join(dir_tmp)
            self.cfg = os.path.join(self.dir_cfg, 'cfg')
            self.copy_scp = os.path.join(self.dir_cfg, 'copy.scp')

            # Open file stream.
            with open(self.copy_scp, 'w') as copy_scp:
                for file_id in self.id_list:
                    wav_file = os.path.join(self.dir_wav, file_id + '.wav')
                    mfc_file = os.path.join(self.dir_mfcc, file_id + '.mfc')
                    os.makedirs(os.path.dirname(mfc_file), exist_ok=True)

                    # print(wav_file + ": " + str(os.path.exists(wav_file)))
                    if os.path.exists(wav_file):
                        # print('{0} {1}\n'.format(wav_file, mfc_file))
                        copy_scp.write('{0} {1}\n'.format(wav_file, mfc_file))
                copy_scp.close()

            if config_file is None:
                # write a CFG for extracting MFCCs
                open(self.cfg, 'w').write(DEFAULT_CONFIG)
            else:
                self.cfg = config_file

            # call htk.
            # print("Using " + self.cfg)
            check_call([HCopy, '-C', self.cfg, '-S', self.copy_scp])


def main():    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-w", "--dir_wav", help="Directory containing the wav files.", type=str,
                        dest="dir_wav", required=True)
    parser.add_argument("-m", "--dir_mfcc", help="Target directory for mfcc files.", type=str,
                        dest="dir_mfcc", required=True)
    parser.add_argument("-f", "--file_id_list", help="Full path to file containing the ids.", type=str,
                        dest="file_id_list", required=True)
    parser.add_argument("-c", "--config_file", help="File used as config for the _HCopy function of htk/hts.", type=str,
                        dest="config_file", required=False)
    
    # Parse arguments
    args = parser.parse_args()
        
    # Read which files to process.
    with open(args.file_id_list) as f:
        id_list = f.readlines()
    # Trim entries in-place.
    id_list[:] = [s.strip(' \t\n\r') for s in id_list]

    # Check for explicit config file as parameter.
    config_file = args.config_file if hasattr(args, 'config_file') else None
    
    # Main functionality.
    print("Force align labels")
    gen_mfcc = GenMFCC(args.dir_wav, args.dir_mfcc, id_list)
    gen_mfcc._HCopy(config_file)


if __name__ == "__main__":
    main()
