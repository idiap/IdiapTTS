#!/usr/bin/env python3
# Scripts taken from Merlin and adapted.
# https://github.com/CSTR-Edinburgh/merlin
# TODO: This file needs cleanup.

# System imports.
import argparse
import logging
import os
import re
import sys
from collections import OrderedDict
import matplotlib.mlab as mlab
import numpy

# Third-party imports.

# Local source tree imports.
if not any("IdiapTTS'" in p for p in sys.path):
    parent_dirs = os.path.realpath(__file__).split(os.sep)
    dir_itts = str.join(os.sep, parent_dirs[:parent_dirs.index("IdiapTTS")+1])
    sys.path.append(dir_itts)  # Adds the IdiapTTS folder to the path, required to work on grid.
from misc.normalisation.MinMaxExtractor import MinMaxExtractor


class LabelNormalisation(object):

    logger = logging.getLogger(__name__)
    htk_label_extension = ".lab"
    questions_label_extension = ".questions"
    file_name_mean_std = "-mean_std_dev"
    file_name_min_max = "-min_max"

    def __init__(self, file_questions):
        self.file_questions = file_questions

    def perform_normalisation(self, file_id_list, id_list, dir_labels, dir_out, return_dict=False):

        self.logger.info('perform linguistic feature extraction')

        normParamsExtractor = MinMaxExtractor()

        if return_dict:
            dict_labels = OrderedDict()

        for file_id in id_list:
            self.logger.debug("Create question labels for " + file_id)
            labels = self.extract_linguistic_features(os.path.join(dir_labels, file_id + self.htk_label_extension),
                                                      os.path.join(dir_out, file_id + self.questions_label_extension))
            normParamsExtractor.add_sample(labels)

            if return_dict:
                dict_labels[file_id] = labels

        file_id_list_name = os.path.splitext(os.path.basename(file_id_list))[0]
        normParamsExtractor.save(os.path.join(dir_out, file_id_list_name))

        # def _perform_normalisation(i):
        #     lab_file = os.path.join(dir_labels, file_id + self.htk_label_extension)
        #     if not dur_file_list:
        #         self.extract_linguistic_features(lab_file, lab_file, label_type)
        #     else:
        #         out_file = os.path.join(dir_out, file_id + self.questions_label_extension)
        #         self.extract_linguistic_features(lab_file, out_file, label_type, dur_file_list[i])
        # pool = Pool()
        # self.utterance_num = len(id_list)
        # pool.map(_perform_normalisation, range(self.utterance_num))
        # pool.close()
        # pool.join()

        norm_params = normParamsExtractor.get_params()
        if return_dict:
            return dict_labels, norm_params
        else:
            return norm_params

    def extract_linguistic_features(self, file_id, out_file_name=None, label_type="state_align", dur_file_name=None):
        if label_type == "phone_align":
            labels = self.load_labels_with_phone_alignment(file_id, dur_file_name)
        elif label_type == "state_align":
            labels = self.load_labels_with_state_alignment(file_id)
        else:
            self.logger.critical("we don't support %s labels as of now!!" % (label_type))

        if out_file_name:
            labels = numpy.array(labels, numpy.float32)
            # numpy.savetxt(out_file_name, labels)
            # path = os.path.dirname(out_file_name)
            # if not os.path.exists(path):
            #     os.makedirs(path)
            with open(out_file_name, 'w') as file:
                labels.tofile(file)

        return labels


class HTSLabelNormalisation(LabelNormalisation):
    """This class is to convert HTS format labels into continous or binary values, and store as binary format with float32 precision.

    The class supports two kinds of questions: QS and CQS.
        **QS**: is the same as that used in HTS

        **CQS**: is the new defined question in the system.  Here is an example of the question: CQS C-Syl-Tone {_(\d+)+}. regular expression is used for continous values.

    Time alignments are expected in the HTS labels. Here is an example of the HTS labels:

    3050000 3100000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[2]

    3100000 3150000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[3]

    3150000 3250000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[4]

    3250000 3350000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[5]

    3350000 3900000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[6]

    305000 310000 are the starting and ending time.
    [2], [3], [4], [5], [6] mean the HMM state index.

    """

    # this subclass support HTS labels, which include time alignments

    def __init__(self, file_questions=None, add_frame_features=True, subphone_feats='full', continuous_flag=True):

        self.question_dict = {}
        self.ori_question_dict = {}
        self.dict_size = 0
        self.continuous_flag = continuous_flag
        try:
#            self.question_dict, self.ori_question_dict = self.load_question_set(question_file_name)
            self.discrete_dict, self.continuous_dict = self.load_question_set_continous(file_questions)
        except:
            self.logger.critical('error whilst loading HTS question set')
            raise

        ###self.dict_size = len(self.question_dict)

        self.dict_size = len(self.discrete_dict) + len(self.continuous_dict)
        self.add_frame_features = add_frame_features
        self.subphone_feats = subphone_feats

        if self.subphone_feats == 'full':
            self.frame_feature_size = 9   ## zhizheng's original 5 state features + 4 phoneme features
        elif self.subphone_feats == 'minimal_frame':
            self.frame_feature_size = 2   ## the minimal features necessary to go from a state-level to frame-level model
        elif self.subphone_feats == 'state_only':
            self.frame_feature_size = 1   ## this is equivalent to a state-based system
        elif self.subphone_feats == 'none':
            self.frame_feature_size = 0   ## the phoneme level features only
        elif self.subphone_feats == 'frame_only':
            self.frame_feature_size = 1   ## this is equivalent to a frame-based system without relying on state-features
        elif self.subphone_feats == 'uniform_state':
            self.frame_feature_size = 2   ## this is equivalent to a frame-based system with uniform state-features
        elif self.subphone_feats == 'minimal_phoneme':
            self.frame_feature_size = 3   ## this is equivalent to a frame-based system with minimal features
        elif self.subphone_feats == 'coarse_coding':
            self.frame_feature_size = 4   ## this is equivalent to a frame-based positioning system reported in Heiga Zen's work
            self.cc_features = self.compute_coarse_coding_features(3)
        else:
            sys.exit('Unknown value for subphone_feats: %s'%(subphone_feats))

        self.dimension = self.dict_size + self.frame_feature_size

        ### if user wants to define their own input, simply set the question set to empty.
        if self.dict_size == 0:
            self.dimension = 0

        self.logger.debug('HTS-derived input feature dimension is %d + %d = %d' % (self.dict_size, self.frame_feature_size, self.dimension) )

    def prepare_dur_data(self, ori_file_list, output_file_list, label_type="state_align", feature_type=None, unit_size=None, feat_size=None):
        '''
        extracting duration binary features or numerical features.
        '''
        logger = logging.getLogger("dur")
        utt_number = len(ori_file_list)
        if utt_number != len(output_file_list):
            print("the number of input and output files should be the same!\n");
            sys.exit(1)

        ### set default feature type to numerical, if not assigned ###
        if not feature_type:
            feature_type = "numerical"

        ### set default unit size to state, if not assigned ###
        if not unit_size:
            unit_size = "state"
        if label_type=="phone_align":
            unit_size = "phoneme"

        ### set default feat size to frame or phoneme, if not assigned ###
        if feature_type=="binary":
            if not feat_size:
                feat_size = "frame"
        elif feature_type=="numerical":
            if not feat_size:
                feat_size = "phoneme"
        else:
            logger.critical("Unknown feature type: %s \n Please use one of the following: binary, numerical\n" %(feature_type))
            sys.exit(1)

        for i in range(utt_number):
            self.extract_dur_features(ori_file_list[i], output_file_list[i], label_type, feature_type, unit_size, feat_size)

    def extract_dur_features(self, in_file_name, out_file_name=None, label_type="state_align", feature_type=None, unit_size=None, feat_size=None):
        logger = logging.getLogger("dur")
        if label_type=="phone_align":
            A = self.extract_dur_from_phone_alignment_labels(in_file_name, feature_type, unit_size, feat_size)
        elif label_type=="state_align":
            A = self.extract_dur_from_state_alignment_labels(in_file_name, feature_type, unit_size, feat_size)
        else:
            logger.critical("we don't support %s labels as of now!!" % (label_type))
            sys.exit(1)

        # if out_file_name:
        #     io_funcs = BinaryIOCollection()
        #     io_funcs.array_to_binary_file(A, out_file_name)
        # else:
        return A

    def extract_dur_from_state_alignment_labels(self, file_name, feature_type, unit_size, feat_size):
        logger = logging.getLogger("dur")

        state_number = 5
        dur_dim = state_number

        if feature_type=="binary":
            dur_feature_matrix = numpy.empty((100000, 1))
        elif feature_type=="numerical":
            if unit_size=="state":
                dur_feature_matrix = numpy.empty((100000, dur_dim))
                current_dur_array = numpy.zeros((dur_dim, 1))
            else: ## phoneme/syllable/word
                dur_feature_matrix = numpy.empty((100000, 1))

        fid = open(file_name)
        utt_labels = fid.readlines()
        fid.close()

        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (file_name, label_number) )

        MLU_dur = [[],[],[]]
        list_of_silences=['#', 'sil', 'pau', 'SIL']
        current_index = 0
        dur_feature_index = 0
        syllable_duration = 0
        word_duration = 0
        for line in utt_labels:
            line = line.strip()

            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            start_time = int(temp_list[0])
            end_time = int(temp_list[1])

            full_label = temp_list[2]
            full_label_length = len(full_label) - 3  # remove state information [k]
            state_index = full_label[full_label_length + 1]
            state_index = int(state_index) - 1
            current_phone = full_label[full_label.index('-') + 1:full_label.index('+')]

            frame_number = int((end_time - start_time)/50000)

            if state_index == 1:
                phone_duration = frame_number

                for i in range(state_number - 1):
                    line = utt_labels[current_index + i + 1].strip()
                    temp_list = re.split('\s+', line)
                    phone_duration += int((int(temp_list[1]) - int(temp_list[0]))/50000)
                
                syllable_duration+=phone_duration
                word_duration+=phone_duration

                ### for syllable and word positional information ###
                label_binary_vector = self.pattern_matching_binary(full_label)
                label_continuous_vector = self.pattern_matching_continous_position(full_label)

                ### syllable ending information ###
                syl_end = 0        
                if(label_continuous_vector[0, 1]==1 or current_phone in list_of_silences): ##pos-bw and c-silences
                    syl_end = 1

                ### word ending information ###
                word_end = 0        
                if(syl_end and label_continuous_vector[0, 9]==1 or current_phone in list_of_silences):
                    word_end = 1

            if feature_type == "binary":
                current_block_array = numpy.zeros((frame_number, 1))
                if unit_size == "state":
                    current_block_array[-1] = 1
                elif unit_size == "phoneme":
                    if state_index == state_number:
                        current_block_array[-1] = 1
                else:
                    logger.critical("Unknown unit size: %s \n Please use one of the following: state, phoneme\n" %(unit_size))
                    sys.exit(1)
            elif feature_type == "numerical":
                if unit_size == "state":
                    current_dur_array[current_index%5] = frame_number
                    if feat_size == "phoneme" and state_index == state_number:
                        current_block_array =  current_dur_array.transpose()
                    if feat_size == "frame":
                        current_block_array = numpy.tile(current_dur_array.transpose(), (frame_number, 1))
                elif state_index == state_number: 
                    if unit_size == "phoneme":
                        current_block_array = numpy.array([phone_duration])
                    elif unit_size == "syllable":
                        current_block_array = numpy.array([syllable_duration])
                    elif unit_size == "word":
                        current_block_array = numpy.array([word_duration])
                    if syl_end:
                        syllable_duration = 0
                    if word_end:
                        word_duration = 0

            ### writing into dur_feature_matrix ###
            if feat_size == "frame":
                dur_feature_matrix[dur_feature_index:dur_feature_index+frame_number,] = current_block_array
                dur_feature_index = dur_feature_index + frame_number
            elif state_index == state_number:
                if feat_size == "phoneme":
                    dur_feature_matrix[dur_feature_index:dur_feature_index+1,] = current_block_array
                    dur_feature_index = dur_feature_index + 1
                elif current_phone!='#': ## removing silence here
                    if feat_size == "syllable" and syl_end:
                        dur_feature_matrix[dur_feature_index:dur_feature_index+1,] = current_block_array
                        dur_feature_index = dur_feature_index + 1
                    elif feat_size == "word" and word_end:
                        dur_feature_matrix[dur_feature_index:dur_feature_index+1,] = current_block_array
                        dur_feature_index = dur_feature_index + 1
                    elif feat_size == "MLU":
                        if word_end:
                            if current_phone=='pau':
                                MLU_dur[0].append(1)
                            else:
                                MLU_dur[0].append(int(label_continuous_vector[0, 24]))
                        if syl_end:
                            if current_phone=='pau':
                                MLU_dur[1].append(1)
                            else:
                                MLU_dur[1].append(int(label_continuous_vector[0, 7]))
                        MLU_dur[2].append(int(phone_duration))


            current_index += 1

        if feat_size == "MLU":
            for seg_indx in xrange(len(MLU_dur)):
                seg_len = len(MLU_dur[seg_indx])
                current_block_array = numpy.reshape(numpy.array(MLU_dur[seg_indx]), (-1, 1))
                dur_feature_matrix[dur_feature_index:dur_feature_index+seg_len, ] = current_block_array
                dur_feature_index = dur_feature_index + seg_len
        
        dur_feature_matrix = dur_feature_matrix[0:dur_feature_index,]
        logger.debug('made duration matrix of %d frames x %d features' % dur_feature_matrix.shape )
        return  dur_feature_matrix

    def extract_dur_from_phone_alignment_labels(self, file_name, feature_type, unit_size, feat_size):
        logger = logging.getLogger("dur")

        dur_dim = 1 # hard coded here 

        if feature_type=="binary":
            dur_feature_matrix = numpy.empty((100000, dur_dim))
        elif feature_type=="numerical":
            if unit_size=="phoneme":
                dur_feature_matrix = numpy.empty((100000, dur_dim))

        fid = open(file_name)
        utt_labels = fid.readlines()
        fid.close()

        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (file_name, label_number) )

        current_index = 0
        dur_feature_index = 0
        for line in utt_labels:
            line = line.strip()

            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            start_time = int(temp_list[0])
            end_time = int(temp_list[1])

            full_label = temp_list[2]

            frame_number = int((end_time - start_time)/50000)

            phone_duration = frame_number

            if feature_type == "binary":
                current_block_array = numpy.zeros((frame_number, 1))
                if unit_size == "phoneme":
                    current_block_array[-1] = 1
                else:
                    logger.critical("Unknown unit size: %s \n Please use one of the following: phoneme\n" %(unit_size))
                    sys.exit(1)
            elif feature_type == "numerical":
                if unit_size == "phoneme":
                    current_block_array = numpy.array([phone_duration])

            ### writing into dur_feature_matrix ###
            if feat_size == "frame":
                dur_feature_matrix[dur_feature_index:dur_feature_index+frame_number,] = current_block_array
                dur_feature_index = dur_feature_index + frame_number
            elif feat_size == "phoneme":
                dur_feature_matrix[dur_feature_index:dur_feature_index+1,] = current_block_array
                dur_feature_index = dur_feature_index + 1

            current_index += 1

        dur_feature_matrix = dur_feature_matrix[0:dur_feature_index,]
        logger.debug('made duration matrix of %d frames x %d features' % dur_feature_matrix.shape )
        return  dur_feature_matrix

    def load_labels_with_phone_alignment(self, file_name, dur_file_name):

        # this is not currently used ??? -- it works now :D
        logger = logging.getLogger("labels")
        #logger.critical('unused function ???')
        #raise Exception

        # if dur_file_name:
        #     io_funcs = BinaryIOCollection()
        #     dur_dim = 1 ## hard coded for now
        #     manual_dur_data = io_funcs.load_binary_file(dur_file_name, dur_dim)

        if self.add_frame_features:
            assert self.dimension == self.dict_size+self.frame_feature_size
        elif self.subphone_feats != 'none':
            assert self.dimension == self.dict_size+self.frame_feature_size
        else:
            assert self.dimension == self.dict_size

        label_feature_matrix = numpy.empty((100000, self.dimension))

        ph_count=0
        label_feature_index = 0
        with open(file_name) as fid:
            all_data = fid.readlines()
        for line in all_data:
            line = line.strip()
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            
            if len(temp_list)==1:
                frame_number = 0
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                full_label = temp_list[2]

                # to do - support different frame shift - currently hardwired to 5msec
                # currently under beta testing: support different frame shift
                if dur_file_name:
                    frame_number = manual_dur_data[ph_count]
                else:
                    frame_number = int((end_time - start_time)/50000)

                if self.subphone_feats == "coarse_coding":
                    cc_feat_matrix = self.extract_coarse_coding_features_relative(frame_number)

            ph_count = ph_count+1
            #label_binary_vector = self.pattern_matching(full_label)
            label_binary_vector = self.pattern_matching_binary(full_label)

            # if there is no CQS question, the label_continuous_vector will become to empty
            label_continuous_vector = self.pattern_matching_continous_position(full_label)
            label_vector = numpy.concatenate([label_binary_vector, label_continuous_vector], axis = 1)

            if self.add_frame_features:
                current_block_binary_array = numpy.zeros((frame_number, self.dict_size+self.frame_feature_size))
                for i in range(frame_number):
                    current_block_binary_array[i, 0:self.dict_size] = label_vector

                    if self.subphone_feats == 'minimal_phoneme':
                        ## features which distinguish frame position in phoneme
                        current_block_binary_array[i, self.dict_size] = float(i+1)/float(frame_number) # fraction through phone forwards
                        current_block_binary_array[i, self.dict_size+1] = float(frame_number - i)/float(frame_number) # fraction through phone backwards
                        current_block_binary_array[i, self.dict_size+2] = float(frame_number) # phone duration

                    elif self.subphone_feats == 'coarse_coding':
                        ## features which distinguish frame position in phoneme using three continous numerical features
                        current_block_binary_array[i, self.dict_size+0] = cc_feat_matrix[i, 0]
                        current_block_binary_array[i, self.dict_size+1] = cc_feat_matrix[i, 1]
                        current_block_binary_array[i, self.dict_size+2] = cc_feat_matrix[i, 2]
                        current_block_binary_array[i, self.dict_size+3] = float(frame_number)

                    elif self.subphone_feats == 'none':
                        pass

                    else:
                        sys.exit('unknown subphone_feats type')

                label_feature_matrix[label_feature_index:label_feature_index+frame_number,] = current_block_binary_array
                label_feature_index = label_feature_index + frame_number

            elif self.subphone_feats == 'none':
                current_block_binary_array = label_vector
                label_feature_matrix[label_feature_index:label_feature_index+1,] = current_block_binary_array
                label_feature_index = label_feature_index + 1

        label_feature_matrix = label_feature_matrix[0:label_feature_index,]

        logger.info('loaded %s, %3d labels' % (file_name, ph_count) )
        logger.debug('made label matrix of %d frames x %d labels' % label_feature_matrix.shape )
        return  label_feature_matrix


    def load_labels_with_state_alignment(self, file_name):
        ## setting add_frame_features to False performs either state/phoneme level normalisation

        if self.add_frame_features:
            assert self.dimension == self.dict_size+self.frame_feature_size
        elif self.subphone_feats != 'none':
            assert self.dimension == self.dict_size+self.frame_feature_size
        else:
            assert self.dimension == self.dict_size

        # label_feature_matrix = numpy.empty((100000, self.dict_size+self.frame_feature_size))
        label_feature_matrix = numpy.empty((100000, self.dimension))

        label_feature_index = 0

        state_number = 5

        lab_binary_vector = numpy.zeros((1, self.dict_size))
        fid = open(file_name)
        utt_labels = fid.readlines()
        fid.close()
        current_index = 0
        label_number = len(utt_labels)
        self.logger.info('loaded %s, %3d labels' % (file_name, label_number) )

        phone_duration = 0
        state_duration_base = 0
        for line in utt_labels:
            line = line.strip()

            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)

            if len(temp_list)==1:
                frame_number = 0
                state_index = 1
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                frame_number = int((end_time - start_time)/50000)  # TODO: Frame size should not be hardcoded.
                full_label = temp_list[2]
            
                full_label_length = len(full_label) - 3  # remove state information [k]
                state_index = full_label[full_label_length + 1]

                state_index = int(state_index) - 1
                state_index_backward = 6 - state_index
                full_label = full_label[0:full_label_length]

            if state_index == 1:
                current_frame_number = 0
                phone_duration = frame_number
                state_duration_base = 0

#                label_binary_vector = self.pattern_matching(full_label)
                label_binary_vector = self.pattern_matching_binary(full_label)

                # if there is no CQS question, the label_continuous_vector will become to empty
                label_continuous_vector = self.pattern_matching_continous_position(full_label)
                label_vector = numpy.concatenate([label_binary_vector, label_continuous_vector], axis = 1)

                if len(temp_list)==1:
                    state_index = state_number
                else:
                    for i in range(state_number - 1):
                        line = utt_labels[current_index + i + 1].strip()
                        temp_list = re.split('\s+', line)
                        phone_duration += int((int(temp_list[1]) - int(temp_list[0]))/50000)

                    if self.subphone_feats == "coarse_coding":
                        cc_feat_matrix = self.extract_coarse_coding_features_relative(phone_duration)

            if self.add_frame_features:
                current_block_binary_array = numpy.zeros((frame_number, self.dict_size+self.frame_feature_size))
                for i in range(frame_number):
                    current_block_binary_array[i, 0:self.dict_size] = label_vector

                    if self.subphone_feats == 'full':
                        ## Zhizheng's original 9 subphone features:
                        current_block_binary_array[i, self.dict_size] = float(i+1) / float(frame_number)   ## fraction through state (forwards)
                        current_block_binary_array[i, self.dict_size+1] = float(frame_number - i) / float(frame_number)  ## fraction through state (backwards)
                        current_block_binary_array[i, self.dict_size+2] = float(frame_number)  ## length of state in frames
                        current_block_binary_array[i, self.dict_size+3] = float(state_index)   ## state index (counting forwards)
                        current_block_binary_array[i, self.dict_size+4] = float(state_index_backward) ## state index (counting backwards)

                        current_block_binary_array[i, self.dict_size+5] = float(phone_duration)   ## length of phone in frames
                        current_block_binary_array[i, self.dict_size+6] = float(frame_number) / float(phone_duration)   ## fraction of the phone made up by current state
                        current_block_binary_array[i, self.dict_size+7] = float(phone_duration - i - state_duration_base) / float(phone_duration) ## fraction through phone (backwards)
                        current_block_binary_array[i, self.dict_size+8] = float(state_duration_base + i + 1) / float(phone_duration)  ## fraction through phone (forwards)

                    elif self.subphone_feats == 'state_only':
                        ## features which only distinguish state:
                        current_block_binary_array[i, self.dict_size] = float(state_index)   ## state index (counting forwards)

                    elif self.subphone_feats == 'frame_only':
                        ## features which distinguish frame position in phoneme:
                        current_frame_number += 1
                        current_block_binary_array[i, self.dict_size] = float(current_frame_number) / float(phone_duration)   ## fraction through phone (counting forwards)

                    elif self.subphone_feats == 'uniform_state':
                        ## features which distinguish frame position in phoneme:
                        current_frame_number += 1
                        current_block_binary_array[i, self.dict_size] = float(current_frame_number) / float(phone_duration)   ## fraction through phone (counting forwards)
                        new_state_index = max(1, round(float(current_frame_number)/float(phone_duration)*5))
                        current_block_binary_array[i, self.dict_size+1] = float(new_state_index)   ## state index (counting forwards)

                    elif self.subphone_feats == "coarse_coding":
                        ## features which distinguish frame position in phoneme using three continous numerical features
                        current_block_binary_array[i, self.dict_size+0] = cc_feat_matrix[current_frame_number, 0]
                        current_block_binary_array[i, self.dict_size+1] = cc_feat_matrix[current_frame_number, 1]
                        current_block_binary_array[i, self.dict_size+2] = cc_feat_matrix[current_frame_number, 2]
                        current_block_binary_array[i, self.dict_size+3] = float(phone_duration)
                        current_frame_number += 1

                    elif self.subphone_feats == 'minimal_frame':
                        ## features which distinguish state and minimally frame position in state:
                        current_block_binary_array[i, self.dict_size] = float(i+1) / float(frame_number)   ## fraction through state (forwards)
                        current_block_binary_array[i, self.dict_size+1] = float(state_index)   ## state index (counting forwards)
                    elif self.subphone_feats == 'none':
                        pass
                    else:
                        sys.exit('unknown subphone_feats type')

                label_feature_matrix[label_feature_index:label_feature_index+frame_number,] = current_block_binary_array
                label_feature_index = label_feature_index + frame_number
            elif self.subphone_feats == 'state_only' and state_index == state_number:
                current_block_binary_array = numpy.zeros((state_number, self.dict_size+self.frame_feature_size))
                for i in range(state_number):
                    current_block_binary_array[i, 0:self.dict_size] = label_vector
                    current_block_binary_array[i, self.dict_size] = float(i+1)   ## state index (counting forwards)
                label_feature_matrix[label_feature_index:label_feature_index+state_number,] = current_block_binary_array
                label_feature_index = label_feature_index + state_number
            elif self.subphone_feats == 'none' and state_index == state_number:
                current_block_binary_array = label_vector
                label_feature_matrix[label_feature_index:label_feature_index+1,] = current_block_binary_array
                label_feature_index = label_feature_index + 1

            state_duration_base += frame_number

            current_index += 1

        label_feature_matrix = label_feature_matrix[0:label_feature_index,]
        self.logger.debug('made label matrix of %d frames x %d labels' % label_feature_matrix.shape )
        return  label_feature_matrix

    def extract_durational_features(self, dur_file_name=None, dur_data=None):

        # if dur_file_name:
        #     io_funcs = BinaryIOCollection()
        #     dur_dim = 1 ## hard coded for now
        #     dur_data = io_funcs.load_binary_file(dur_file_name, dur_dim)

        ph_count = len(dur_data)
        total_num_of_frames = int(sum(dur_data))

        duration_feature_array = numpy.zeros((total_num_of_frames, self.frame_feature_size))

        frame_index=0
        for i in range(ph_count):
            frame_number = int(dur_data[i])
            if self.subphone_feats == "coarse_coding":
                cc_feat_matrix = self.extract_coarse_coding_features_relative(frame_number)

                for j in range(frame_number):
                    duration_feature_array[frame_index, 0] = cc_feat_matrix[j, 0]
                    duration_feature_array[frame_index, 1] = cc_feat_matrix[j, 1]
                    duration_feature_array[frame_index, 2] = cc_feat_matrix[j, 2]
                    duration_feature_array[frame_index, 3] = float(frame_number)
                    frame_index+=1

            elif self.subphone_feats == 'full':
                state_number = 5 # hard coded here 
                phone_duration = sum(dur_data[i, :])
                state_duration_base = 0
                for state_index in xrange(1, state_number+1):
                    state_index_backward = (state_number - state_index) + 1
                    frame_number = int(dur_data[i][state_index-1])
                    for j in xrange(frame_number):
                        duration_feature_array[frame_index, 0] = float(j+1) / float(frame_number)   ## fraction through state (forwards)
                        duration_feature_array[frame_index, 1] = float(frame_number - j) / float(frame_number)  ## fraction through state (backwards)
                        duration_feature_array[frame_index, 2] = float(frame_number)  ## length of state in frames
                        duration_feature_array[frame_index, 3] = float(state_index)   ## state index (counting forwards)
                        duration_feature_array[frame_index, 4] = float(state_index_backward) ## state index (counting backwards)
    
                        duration_feature_array[frame_index, 5] = float(phone_duration)   ## length of phone in frames
                        duration_feature_array[frame_index, 6] = float(frame_number) / float(phone_duration)   ## fraction of the phone made up by current state
                        duration_feature_array[frame_index, 7] = float(phone_duration - j - state_duration_base) / float(phone_duration) ## fraction through phone (forwards)
                        duration_feature_array[frame_index, 8] = float(state_duration_base + j + 1) / float(phone_duration)  ## fraction through phone (backwards)
                        frame_index+=1
                    
                    state_duration_base += frame_number

        return duration_feature_array

    def compute_coarse_coding_features(self, num_states):
        assert num_states == 3

        npoints = 600
        cc_features = numpy.zeros((num_states, npoints))

        x1 = numpy.linspace(-1.5, 1.5, npoints)
        x2 = numpy.linspace(-1.0, 2.0, npoints)
        x3 = numpy.linspace(-0.5, 2.5, npoints)

        mu1 = 0.0
        mu2 = 0.5
        mu3 = 1.0

        sigma = 0.4

        cc_features[0, :] = mlab.normpdf(x1, mu1, sigma)
        cc_features[1, :] = mlab.normpdf(x2, mu2, sigma)
        cc_features[2, :] = mlab.normpdf(x3, mu3, sigma)

        return cc_features

    def extract_coarse_coding_features_relative(self, phone_duration):
        dur = int(phone_duration)

        cc_feat_matrix = numpy.zeros((dur, 3))

        for i in range(dur):
            rel_indx = int((200/float(dur))*i)
            cc_feat_matrix[i,0] = self.cc_features[0, 300+rel_indx]
            cc_feat_matrix[i,1] = self.cc_features[1, 200+rel_indx]
            cc_feat_matrix[i,2] = self.cc_features[2, 100+rel_indx]

        return cc_feat_matrix


    def pattern_matching_binary(self, label):

        dict_size = len(self.discrete_dict)
        lab_binary_vector = numpy.zeros((1, dict_size))

        for i in range(dict_size):
            current_question_list = self.discrete_dict[str(i)]
            binary_flag = 0
            for iq in range(len(current_question_list)):
                current_compiled = current_question_list[iq]

                ms = current_compiled.search(label)
                if ms is not None:
                    binary_flag = 1
                    break
            lab_binary_vector[0, i] = binary_flag

        return   lab_binary_vector


    def pattern_matching_continous_position(self, label):

        dict_size = len(self.continuous_dict)

        lab_continuous_vector = numpy.zeros((1, dict_size))

        for i in range(dict_size):
            continuous_value = -1.0

            current_compiled = self.continuous_dict[str(i)]

            ms = current_compiled.search(label)
            if ms is not None:
#                assert len(ms.group()) == 1
                continuous_value = ms.group(1)

            lab_continuous_vector[0, i] = continuous_value

        return  lab_continuous_vector

    def load_question_set(self, qs_file_name):
        fid = open(qs_file_name)
        question_index = 0
        question_dict = {}
        ori_question_dict = {}
        for line in fid.readlines():
            line = line.replace('\n', '')
            if len(line) > 5:
                temp_list = line.split('{')
                temp_line = temp_list[1]
                temp_list = temp_line.split('}')
                temp_line = temp_list[0]
                question_list = temp_line.split(',')
                question_dict[str(question_index)] = question_list
                ori_question_dict[str(question_index)] = line
                question_index += 1
        fid.close()

        logger = logging.getLogger("labels")
        logger.debug('loaded question set with %d questions' % len(question_dict))

        return  question_dict, ori_question_dict


    def load_question_set_continous(self, qs_file_name):

        logger = logging.getLogger("labels")

        fid = open(qs_file_name)
        binary_qs_index = 0
        continuous_qs_index = 0
        binary_dict = {}
        continuous_dict = {}
        LL=re.compile(re.escape('LL-'))

        for line in fid.readlines():
            line = line.replace('\n', '')

            if len(line) > 5:
                temp_list = line.split('{')
                temp_line = temp_list[1]
                temp_list = temp_line.split('}')
                temp_line = temp_list[0]
                temp_line = temp_line.strip()
                question_list = temp_line.split(',')

                temp_list = line.split(' ')
                question_key = temp_list[1]
#                print   line
                if temp_list[0] == 'CQS':
                    assert len(question_list) == 1
                    processed_question = self.wildcards2regex(question_list[0], convert_number_pattern=True)
                    continuous_dict[str(continuous_qs_index)] = re.compile(processed_question) #save pre-compiled regular expression
                    continuous_qs_index = continuous_qs_index + 1
                elif temp_list[0] == 'QS':
                    re_list = []
                    for temp_question in question_list:
                        processed_question = self.wildcards2regex(temp_question)
                        if LL.search(question_key):
                            processed_question = '^'+processed_question
                        re_list.append(re.compile(processed_question))

                    binary_dict[str(binary_qs_index)] = re_list
                    binary_qs_index = binary_qs_index + 1
                else:
                    logger.critical('The question set is not defined correctly: %s' %(line))
                    raise Exception

#                question_index = question_index + 1
        return  binary_dict, continuous_dict


    def wildcards2regex(self, question, convert_number_pattern=False):
        """
        Convert HTK-style question into regular expression for searching labels.
        If convert_number_pattern, keep the following sequences unescaped for
        extracting continuous values):
            (\d+)       -- handles digit without decimal point
            ([\d\.]+)   -- handles digits with and without decimal point
        """

        ## handle HTK wildcards (and lack of them) at ends of label:
        prefix = ""
        postfix = ""
        if '*' in question:
            if not question.startswith('*'):
                prefix = "\A"
            if not question.endswith('*'):
                postfix = "\Z"
        question = question.strip('*')
        question = re.escape(question)
        ## convert remaining HTK wildcards * and ? to equivalent regex:
        question = question.replace('\\*', '.*')
        question = prefix + question + postfix

        if convert_number_pattern:
            question = question.replace('\\(\\\\d\\+\\)', '(\d+)')
            question = question.replace('\\(\\[\\\\d\\\\\\.\\]\\+\\)', '([\d\.]+)')
        return question






class HTSDurationLabelNormalisation(HTSLabelNormalisation):
    """
    Unlike HTSLabelNormalisation, HTSDurationLabelNormalisation does not accept timings.
    One line of labels is converted into 1 datapoint, that is, the label is not 'unpacked'
    into frames. HTK state index [\d] is not handled in any special way.
    """
    def __init__(self, question_file_name=None, subphone_feats='full', continuous_flag=True):
        super(HTSDurationLabelNormalisation, self).__init__(question_file_name=question_file_name, \
                                    subphone_feats=subphone_feats, continuous_flag=continuous_flag)
        ## don't use extra features beyond those in questions for duration labels:
        self.dimension = self.dict_size


    def load_labels_with_state_alignment(self, file_name, add_frame_features=False):
        ## add_frame_features not used in HTSLabelNormalisation -- only in XML version

        logger = logging.getLogger("labels")

        assert self.dimension == self.dict_size

        label_feature_matrix = numpy.empty((100000, self.dimension))

        label_feature_index = 0


        lab_binary_vector = numpy.zeros((1, self.dict_size))
        fid = open(file_name)
        utt_labels = fid.readlines()
        fid.close()
        current_index = 0
        label_number = len(utt_labels)
        logger.info('loaded %s, %3d labels' % (file_name, label_number) )

        ## remove empty lines
        utt_labels = [line for line in utt_labels if line != '']

        for (line_number, line) in enumerate(utt_labels):
            temp_list = re.split('\s+', line.strip())
            full_label = temp_list[-1]  ## take last entry -- ignore timings if present

            label_binary_vector = self.pattern_matching_binary(full_label)

            # if there is no CQS question, the label_continuous_vector will become to empty
            label_continuous_vector = self.pattern_matching_continous_position(full_label)
            label_vector = numpy.concatenate([label_binary_vector, label_continuous_vector], axis = 1)

            label_feature_matrix[line_number, :] = label_vector[:]


        label_feature_matrix = label_feature_matrix[:line_number+1,:]
        logger.debug('made label matrix of %d frames x %d labels' % label_feature_matrix.shape )
        return  label_feature_matrix


#  -----------------------------


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-l", "--dir_labels", help="Directory containing the HTK labels.", type=str,
                        dest="dir_labels", required=True)
    parser.add_argument("--dir_questions", help="Target directory for question files.", type=str,
                        dest="dir_questions", required=True)
    parser.add_argument("-q", "--file_questions", help="Full path to file containing the questions.", type=str,
                        dest="file_questions", required=True)
    parser.add_argument("-f", "--file_id_list", help="Full path to file containing the ids.", type=str,
                        dest="file_id_list", required=True)
    # parser.add_argument("-c", "--config_file", help="File used as config for the _HCopy function of htk/hts.", type=str,
    #                     dest="config_file", required=False)
    
    # Parse arguments
    args = parser.parse_args()
        
    # Read which files to process.
    print(args.file_id_list)
    with open(args.file_id_list) as f:
        id_list = f.readlines()
    # Trim entries in-place.
    id_list[:] = [s.strip(' \t\n\r') for s in id_list]

    label_operater = HTSLabelNormalisation(args.file_questions)
    label_operater.perform_normalisation(args.file_id_list, id_list, args.dir_labels, args.dir_questions)
    #feature_type="binary"
    #unit_size = "phoneme"
    #feat_size = "phoneme"
    #label_operater.prepare_dur_data(ori_file_list, output_file_list, feature_type, unit_size, feat_size)
    #label_operater.prepare_dur_data(ori_file_list, output_file_list, feature_type)
    #print(label_operater.dimension)
