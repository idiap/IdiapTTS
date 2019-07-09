#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scripts taken from Merlin and adapted.
# https://github.com/CSTR-Edinburgh/merlin

__author__ = "Bastian Schnell"
__version__ = "Revision: 1.0"
__date__ = "Date: 2018/03/29"
__copyright__ = "Copyright (c) 2018 by Idiap Research Institute, http://www.idiap.ch"
__license__ = "licence.txt"

import os, sys
import time
import argparse
import random
import glob
import shutil

from sys import argv, stderr
from subprocess import check_call, Popen, CalledProcessError, PIPE
from mean_variance_norm import MeanVarianceNorm

# string constants for various shell calls
STATE_NUM = 5
F = str(0.01)
SFAC = str(5.0)
PRUNING = [str(i) for i in (250., 150., 5000.)]  # Merlin uses 2000. as max threshold.

MACROS = 'macros'
HMMDEFS = 'hmmdefs'
VFLOORS = 'vFloors'

##
HTKDIR = os.path.abspath("../../../tools/bin/htk/")
HCompV = os.path.join(HTKDIR, 'HCompV')
HCopy  = os.path.join(HTKDIR, 'HCopy' )
HERest = os.path.join(HTKDIR, 'HERest')
HHEd   = os.path.join(HTKDIR, 'HHEd'  )
HVite  = os.path.join(HTKDIR, 'HVite' )

class ForcedAlignment(object):

    def __init__(self):
        self.proto = None
        self.phoneme_mlf = None

    def _make_proto(self):
        ## make proto
        fid = open(self.proto, 'w')
        means = ' '.join(['0.0' for _ in range(39)])
        varg = ' '.join(['1.0' for _ in range(39)])
        fid.write("""~o <VECSIZE> 39 <USER>
~h "proto"
<BEGINHMM>
<NUMSTATES> 7""")
        for i in range(2, STATE_NUM+2):
            fid.write('<STATE> {0}\n<MEAN> 39\n{1}\n'.format(i, means))
            fid.write('<VARIANCE> 39\n{0}\n'.format(varg))
        fid.write("""<TRANSP> 7
 0.0 1.0 0.0 0.0 0.0 0.0 0.0
 0.0 0.6 0.4 0.0 0.0 0.0 0.0
 0.0 0.0 0.6 0.4 0.0 0.0 0.0
 0.0 0.0 0.0 0.6 0.4 0.0 0.0
 0.0 0.0 0.0 0.0 0.6 0.4 0.0
 0.0 0.0 0.0 0.0 0.0 0.7 0.3
 0.0 0.0 0.0 0.0 0.0 0.0 0.0
<ENDHMM>
""")
        fid.close()

        ## make vFloors
        check_call([HCompV, '-f', F, '-C', self.cfg,
                            '-S', self.train_scp,
                            '-M', self.cur_dir, self.proto])
        ## make local macro
        # get first three lines from local proto
        fid = open(os.path.join(self.cur_dir, MACROS), 'w')
        source = open(os.path.join(self.cur_dir,
                      os.path.split(self.proto)[1]), 'r')
        for _ in range(3):
            fid.write(source.readline())
        source.close()
        # get remaining lines from vFloors
        fid.writelines(open(os.path.join(self.cur_dir,
                                         VFLOORS), 'r').readlines())
        fid.close()
        ## make hmmdefs
        fid = open(os.path.join(self.cur_dir, HMMDEFS), 'w')
        for phone in open(self.phonemes, 'r'):
            source = open(self.proto, 'r')
            # ignore
            source.readline()
            source.readline()
            # the header
            fid.write('~h "{0}"\n'.format(phone.rstrip()))
            # the rest
            fid.writelines(source.readlines())
            source.close()
        fid.close()

    def _read_file_list(self, file_name):

        file_lists = []
        fid = open(file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            file_lists.append(line)
        fid.close()

        return file_lists

    def _full_to_mono(self, full_file_name, mono_file_name, phoneme_dict):
        fre = open(full_file_name, 'r')
        fwe = open(mono_file_name, 'w')
        for line in fre.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            tmp_list = line.split('-')
            tmp_list = tmp_list[1].split('+')
            mono_phone = tmp_list[0]
            fwe.write('{0}\n'.format(mono_phone))
            if mono_phone not in phoneme_dict:
                # print("Found new phoneme {}.".format(mono_phone))
                phoneme_dict[mono_phone] = 1
            phoneme_dict[mono_phone] += 1
        fwe.close()
        fre.close()

    def _check_data(self, file_id_list, multiple_speaker):

        copy_scp = open(self.copy_scp, 'w')
        check_scp = open(self.train_scp, 'w')
        i = 0

        phoneme_dict = {}
        speaker_utt_dict = {}

        for file_id in file_id_list:
            wav_file = os.path.join(self.wav_dir, file_id + '.wav')
            lab_file = os.path.join(self.lab_dir, file_id + '.lab')
            mfc_file = os.path.join(self.mfc_dir, file_id + '.mfc')
            mono_lab_file = os.path.join(self.mono_lab_dir, file_id + '.lab')

            mfc_sub_dir = os.path.dirname(mfc_file)
            if not os.path.exists(wav_file):
                print("Error: Audio file does not exist: {}.".format(wav_file))
                # print(wav_file + ": " + str(os.path.exists(wav_file)))
            if not os.path.exists(lab_file):
                print("Error: Label file does not exist: {}.".format(lab_file))
                # print(lab_file + ": " + str(os.path.exists(lab_file)))
            if os.path.exists(wav_file) and os.path.exists(lab_file):
                # print('{0} {1}\n'.format(wav_file, mfc_file))
                if not os.path.exists(mfc_sub_dir):
                    os.makedirs(mfc_sub_dir)

                copy_scp.write('{0} {1}\n'.format(wav_file, mfc_file))
                check_scp.write('{0}\n'.format(mfc_file))

                if multiple_speaker:
                    tmp_list = file_id.split('/')
                    speaker_name = tmp_list[0]
                    if speaker_name not in speaker_utt_dict:
                        speaker_utt_dict[speaker_name] = []
                    speaker_utt_dict[speaker_name].append(mfc_file)
                else:
                    if 'only_one' not in speaker_utt_dict:
                        speaker_utt_dict['only_one'] = []
                    speaker_utt_dict['only_one'].append(mfc_file)

                # print("Add phonemes from {}.".format(file_id))
                self._full_to_mono(lab_file, mono_lab_file, phoneme_dict)
        copy_scp.close()
        check_scp.close()

        fid = open(self.phonemes, 'w')
        fmap = open(self.phoneme_map, 'w')
        for phoneme in list(phoneme_dict.keys()):
            fid.write('{0}\n'.format(phoneme))
            fmap.write('{0} {0}\n'.format(phoneme))
        fmap.close()
        fid.close()

        self.phoneme_mlf = os.path.join(self.cfg_dir, 'mono_phone.mlf')
        fid = open(self.phoneme_mlf, 'w')
        fid.write('#!MLF!#\n')
        fid.write('"*/*.lab" -> "' + self.mono_lab_dir + '"\n')
        fid.close()

        return speaker_utt_dict

    def _nxt_dir(self):
        """
        Get the next HMM directory
        """
        # pass on the previously new one to the old one
        self.cur_dir = self.nxt_dir
        # increment
        self.n += 1
        # compute the path for the new one
        self.nxt_dir = os.path.join(self.hmm_dir, str(self.n).zfill(3))
        # make the new directory
        os.mkdir(self.nxt_dir)

    def prepare_training(self, file_id_list_name, wav_dir, lab_dir, mfc_dir, work_dir, multiple_speaker):

        print('---preparing enverionment')
        self.cfg_dir = os.path.join(work_dir, 'config')
        self.model_dir = os.path.join(work_dir, 'model')
        self.cur_dir = os.path.join(self.model_dir, 'hmm0')
        if not os.path.exists(self.cfg_dir):
            os.makedirs(self.cfg_dir)
        if not os.path.exists(self.cur_dir):
            os.makedirs(self.cur_dir)

        self.phonemes = os.path.join(work_dir, 'mono_phone.list')
        self.phoneme_map = os.path.join(work_dir, 'phoneme_map.dict')
        # HMMs
        self.proto = os.path.join(self.cfg_dir, 'proto')
        # SCP files
        self.copy_scp = os.path.join(self.cfg_dir, 'copy.scp')
        self.test_scp = os.path.join(self.cfg_dir, 'test.scp')
        self.train_scp = os.path.join(self.cfg_dir, 'train.scp')
        # CFG
        self.cfg = os.path.join(self.cfg_dir, 'cfg')
        
        # The following is only correct for the default values in the cfg file given
        # in gen_mfcc.py. If you changed the cfg for mfcc extraction, change it here accordingly.
        open(self.cfg, 'w').write("""TARGETRATE = 50000.0
TARGETKIND = USER
WINDOWSIZE = 250000.0
PREEMCOEF = 0.97
USEHAMMING = T
ENORMALIZE = T
CEPLIFTER = 22
NUMCHANS = 20
NUMCEPS = 12
""")

        self.wav_dir = wav_dir
        self.lab_dir = lab_dir
        self.mfc_dir = mfc_dir
        if not os.path.exists(self.mfc_dir):
            os.makedirs(self.mfc_dir)

        self.mono_lab_dir = os.path.join(work_dir, 'mono_no_align')
        if not os.path.exists(self.mono_lab_dir):
            # shutil.rmtree(self.mono_lab_dir)
            os.makedirs(self.mono_lab_dir)

        file_id_list = self._read_file_list(file_id_list_name)
        print('---checking data')
        speaker_utt_dict = self._check_data(file_id_list, multiple_speaker)
        
        print('---feature_normalisation')
        for key_name in list(speaker_utt_dict.keys()):
            print("   ---create normaliser for speaker {}.".format(key_name))
            normaliser = MeanVarianceNorm(39)
            normaliser.feature_normalisation(speaker_utt_dict[key_name], speaker_utt_dict[key_name])  ## save to itself
        print(time.strftime("%c"))

        print('---making proto')
        self._make_proto()

    def train_hmm(self, niter, num_mix):
        """
        Perform one or more rounds of estimation
        """

        print(time.strftime("%c"))
        print('---training HMM models')

        # call HErest in multiple chunks
        # split scp in num_splits chunks and save them
        num_splits = int(os.getenv('DNN_NUM_PARALLEL', 8))
        print("----num_splits set to %s" % num_splits)
        train_scp_chunks = []
        with open(self.train_scp, "rt") as fp:
            mfc_files = fp.readlines()
        random.shuffle(mfc_files)
        n = int((len(mfc_files) + 1) / num_splits)
        mfc_chunks = [mfc_files[j:j + n] for j in range(0, len(mfc_files), n)]
        for i in range(len(mfc_chunks)):
            train_scp_chunks.append(os.path.join(self.cfg_dir, "train_%d.scp" % i))
            with open(train_scp_chunks[i], "wt") as fp:
                fp.writelines(mfc_chunks[i])

        done = 0
        mix = 1
        while mix <= num_mix and done == 0:
            for i in range(niter):
                next_dir = os.path.join(self.model_dir, 'hmm_mix_' + str(mix) + '_iter_' + str(i + 1))
                if not os.path.exists(next_dir):
                    os.makedirs(next_dir)

                procs = []
                # estimate per chunk
                for chunk_num in range(len(train_scp_chunks)):
                    procs.append(Popen([HERest, '-C', self.cfg,
                                        '-S', train_scp_chunks[chunk_num],
                                        '-I', self.phoneme_mlf,
                                        '-M', next_dir,
                                        '-H', os.path.join(self.cur_dir, MACROS),
                                        '-H', os.path.join(self.cur_dir, HMMDEFS),
                                        '-t'] + PRUNING + ['-p', str(chunk_num + 1), self.phonemes],
                                       stdout=PIPE))
                # wait until all HERest calls are finished
                for p in procs:
                    p.wait()

                # now accumulate
                check_call([HERest, '-C', self.cfg,
                            '-M', next_dir,
                            '-H', os.path.join(self.cur_dir, MACROS),
                            '-H', os.path.join(self.cur_dir, HMMDEFS),
                            '-t'] + PRUNING + ['-p', '0', self.phonemes] + glob.glob(next_dir + os.sep + "*.acc"),
                           stdout=PIPE)

                self.cur_dir = next_dir

            if mix * 2 <= num_mix:
                ##increase mixture number
                hed_file = os.path.join(self.cfg_dir, 'mix_' + str(mix * 2) + '.hed')
                fid = open(hed_file, 'w')
                fid.write('MU ' + str(mix * 2) + ' {*.state[2-'+str(STATE_NUM+2)+'].mix}\n')
                fid.close()

                next_dir = os.path.join(self.model_dir, 'hmm_mix_' + str(mix * 2) + '_iter_0')
                if not os.path.exists(next_dir):
                    os.makedirs(next_dir)

                check_call( [HHEd, '-A',
                             '-H', os.path.join(self.cur_dir, MACROS),
                             '-H', os.path.join(self.cur_dir, HMMDEFS),
                             '-M', next_dir] + [hed_file] + [self.phonemes])

                self.cur_dir = next_dir
                mix = mix * 2
            else:
                done = 1

    def align(self, work_dir, lab_align_dir):
        """
        Align using the models in self.cur_dir and MLF to path
        """
        print('---aligning data')
        print(time.strftime("%c"))
        self.align_mlf = os.path.join(work_dir, 'mono_align.mlf')

        check_call([HVite, '-a', '-f', '-m', '-y', 'lab', '-o', 'SM',
                    '-i', self.align_mlf, '-L', self.mono_lab_dir,
                    '-C', self.cfg, '-S', self.train_scp,
                    '-H', os.path.join(self.cur_dir, MACROS),
                    '-H', os.path.join(self.cur_dir, HMMDEFS),
                    '-I', self.phoneme_mlf, '-t'] + PRUNING +
                   ['-s', SFAC, self.phoneme_map, self.phonemes])

        self._postprocess(self.align_mlf, lab_align_dir)

    def _postprocess(self, mlf, lab_align_dir):
        if not os.path.exists(lab_align_dir):
            os.makedirs(lab_align_dir)

        # fstats = open("logprob.txt", "wt")
        state_num = STATE_NUM
        fid = open(mlf, 'r')
        line = fid.readline()
        while True:
            line = fid.readline()
            line = line.strip()
            if len(line) < 1:
                break
            line = line.replace('"', '')
            file_base = os.path.basename(line)
            flab = open(os.path.join(self.lab_dir, file_base), 'r')
            fw = open(os.path.join(lab_align_dir, file_base), 'w')
            lab_logprob = 0.0
            lab_entries = 0
            for full_lab in flab.readlines():
                full_lab = full_lab.strip()
                for i in range(state_num):
                    line = fid.readline()
                    line = line.strip()
                    tmp_list = line.split()
                    fw.write('{0} {1} {2}[{3}]\n'.format(tmp_list[0], tmp_list[1], full_lab, i + 2))
                    # print(tmp_list)
                    # lab_logprob += float(tmp_list[3])  # Not sure what this should do, but it throws an error.
                    lab_entries += 1

            fw.close()
            flab.close()
            # fstats.write(file_base + " " + str(lab_entries) + " " + str(lab_logprob / lab_entries))
            line = fid.readline()
            line = line.strip()
            if line != '.':
                print('The two files are not matched!\n')
                sys.exit(1)
        fid.close()
        # fstats.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir_work", help="Directory to work in.", type=str,
                        dest="dir_work", required=True)
    parser.add_argument("-w", "--dir_wav", help="Directory containing the wav files.", type=str,
                        dest="dir_wav", required=True)
    parser.add_argument("-l", "--dir_lab", help="Directory containing the htk labels without durations.", type=str,
                        dest="dir_lab", required=True)
    parser.add_argument("-m", "--dir_mfcc", help="Directory containing the mfcc files.", type=str,
                        dest="dir_mfcc", required=True)
    parser.add_argument("-f", "--file_id_list", help="Full path to file containing the ids.", type=str,
                        dest="file_id_list", required=True)
    parser.add_argument("-a", "--dir_lab_align", help="Directory to store the aligned labels in.", type=str,
                        dest="dir_lab_align", required=True)
    parser.add_argument("--multiple_speaker", help="Used to do speaker-dependent normalisation.",
                        dest="multiple_speaker", action='store_const', const=True, default=False)
    # Parse arguments.
    args = parser.parse_args()

    ## if multiple_speaker is tuned on. the file_id_list.scp has to reflact this
    ## for example
    ## speaker_1/0001
    ## speaker_2/0001
    ## This is to do speaker-dependent normalisation
    multiple_speaker = args.multiple_speaker

    aligner = ForcedAlignment()
    aligner.prepare_training(args.file_id_list, args.dir_wav, args.dir_lab, args.dir_mfcc, args.dir_work, multiple_speaker)

    aligner.train_hmm(7, 32)
    aligner.align(args.dir_work, args.dir_lab_align)
    print('---done!')

if __name__ == '__main__':
    main()
