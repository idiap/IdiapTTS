#!/usr/bin/env bash

# Copyright 2018 by Idiap Research Institute, http://www.idiap.ch
#
# Author(s):
#   Bastian Schnell, 04.04.2018
#
# Execution requires: source cmd.sh


usage() {
cat <<- EOF
    usage: ${PROGNAME} [OPTIONS] <voice_name=demo> <num_workers=1> <file_id_list=database/file_id_list_<voice_name>>

    Compute HTK full labels for all ids listed in <file_id_list>.
    Audio files are searched in database/wav/.
    By default ensures a half second of silence in front and back of each file before computing the labels.

    OPTIONS:
        -h                        show this help
        -m, --multispeaker        Set for multispeaker databases.

    
    Examples:
        Run all ids with 10 jobs:
        ${PROGNAME} demo 10 database/file_id_list_full.txt
EOF
}

###############################
# Default options and functions
#
# set -o xtrace # Prints every command before running it, same as "set -x".
# set -o errexit # Exit when a command fails, same as "set -e".
#                # Use "|| true" for those who are allowed to fail.
#                # Disable (set +e) this mode if you want to know a nonzero return value.
# set -o pipefail # Catch mysqldump fails.
# set -o nounset # Exit when using undeclared variables, same as "set -u".
# set -o noclobber # Prevents the bash shell from overwriting files, but you can force it with ">|".
export SHELLOPTS # Used to pass above shell options to any called subscripts.

readonly PROGNAME=$(basename $0)
readonly PROGDIR=$(readlink -m $(dirname $0))
readonly ARGS="$@"

# Fixed paths.
dir_src=$(realpath "../../../src/")
dir_tools=$(realpath "../../../tools/")
dir_misc=$(realpath "../../../misc//")
dir_data=$(realpath "database/")

log()
{
    echo -e >&2 "$@"
}

# Die should be called when an error occurs with a HELPFUL error message.
die () {
    log "ERROR" "$@"
    exit 1
}

multispeaker=""
while getopts ":hm-:" flag; do # If a character is followed by a colon (e.g. f:), that option is expected to have an argument.
        case "${flag}" in
            -) case "${OPTARG}" in
                   multispeaker) multispeaker="--multiple_speaker" ;;
                   *) die "Invalid option: --${OPTARG}" ;;
               esac;;
            h) usage; exit ;;
            m) multispeaker="--multiple_speaker" ;;
            \?) die "Invalid option: -$OPTARG" ;;
            :)  die "Option -$OPTARG requires an argument." ;;
        esac
    done
    shift $(($OPTIND - 1)) # Skip the already processed arguments.

# Read parameters.
voice=${1:-"demo"}
num_workers=${2:-"1"}  # Default number of workers is one.
file_id_list=${3:-"${dir_data}/file_id_list_${voice}.txt"}

# Fixed path with parameters.
dir_labels=$(realpath "experiments")"/${voice}/labels/"
dir_logs="${dir_labels}/log/"
dir_mfcc="${dir_labels}/mfc/"

# Create necessary directories.
mkdir -p ${dir_labels}
mkdir -p ${dir_logs}

# Load utts list.
IFS=$'\r\n' GLOBIGNORE='*' command eval 'utts=($(cat $file_id_list))'
num_utts=${#utts[@]}
block_size=$(expr ${num_utts} / ${num_workers} + 1)
num_blocks=$(expr ${num_utts} / ${block_size} + 1)

echo "Get selected utterances..."
# Combine the selected utterances to a regex pattern.
pat_size=1000
rm -f ${dir_labels}/utts_selected.data
iterations=$(expr ${num_utts} / ${pat_size} )
iterations=$(( 0 > ${iterations} ? 0 : ${iterations} ))
for block in $(eval echo "{0..${iterations}}"); do
    start_index=$(( ${block} * ${pat_size} ))
#    end_index=$(( (${block} + 1) * ${pat_size} ))
#    echo $start_index
    utts_pat=$(echo ${utts[@]:$start_index:$pat_size}|sed 's/ /\\|/g'|sed 's#\/#\\/#g')
#    echo ${utts_pat[@]}
    # Select those labes of utts.data which belong to the selected utterances.
    sed -n "/${utts_pat}/p" ${dir_data}/utts.data >> ${dir_labels}/utts_selected.data
    ##cat ${dir_data}/utts.data | grep -wE "${utts_pat}" >| ${dir_labels}/utts_selected.data
done
sed -i 's/[()]//g' ${dir_labels}/utts_selected.data
sed -i 's/ /'$'\t''/' ${dir_labels}/utts_selected.data  # Convert first space into tab to match expected format of tts_frontend.

# # Create links to all selected .txt files.
# for utt in "${utts[@]}"; do
#     ln -sf ${dir_data}/txt/${utt}.txt ${dir_labels}/${utt}.txt
# done

# Split into working blocks.
if [ "$num_blocks" -gt "99" ]; then
    suffix_length=3
elif [ "$num_blocks" -gt "9" ]; then
    suffix_length=2
else
    suffix_length=1
fi

# Split into sub parts.
split --numeric=1 --suffix-length ${suffix_length} -l ${block_size} ${dir_labels}/utts_selected.data ${dir_labels}/utts_selected.data_block
# Remove leading zeros in sub parts numbers.
for FILE in $(ls ${dir_labels}/utts_selected.data_block*); do
    mv "${FILE}" "$(echo ${FILE} | sed -e 's:_block0*:_block:')" 2>/dev/null
done

# Split file_id_list in the same way as utts_selected.data.
name_file_id_list=$(basename ${file_id_list})
split --numeric=1 --suffix-length ${suffix_length} -l ${block_size} ${file_id_list} ${dir_labels}/${name_file_id_list}_block
# Remove leading zeros in block numbering.
for FILE in $(ls ${dir_labels}/${name_file_id_list}_block*); do
    mv "${FILE}" "$(echo ${FILE} | sed -e 's:_block0*:_block:')" 2>/dev/null
done

# Create labels for all utterance blocks. Send commands as job array.
echo "Create labels..."
./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/utts_selected.dataJOB.log ${dir_tools}/tts_frontend/English/makeLabels.sh ${dir_tools}/festival/ ${dir_labels}/utts_selected.data_blockJOB BR ${dir_labels}/../

# Remove intermediate files.
rm -f ${dir_labels}/utts_selected.data
eval rm -f ${dir_labels}/utts_selected.data_block{0..${num_blocks}}  # eval command required because otherwise brace expansion { .. } happens before $ expansion

# Create labels without duration (no alignment data contained).
echo "Remove durations..."
mkdir -p "${dir_labels}/full_no_align/"
mkdir -p "${dir_labels}/mono_no_align/"
for i in ${dir_labels}/full/*.lab; do cp "$i" ${dir_labels}/full_no_align/; done
find ${dir_labels}/full_no_align/ -name "*lab" -exec sed -i 's/[ ]*[^ ]*[ ]*[^ ]* //' {} +
#sed -i 's/[ ]*[^ ]*[ ]*[^ ]* //' ${dir_labels}/full_no_align/*.lab
for i in ${dir_labels}/mono/*.lab; do cp "$i" ${dir_labels}/mono_no_align/; done
find ${dir_labels}/mono_no_align/ -name "*.lab" -exec sed -i 's/[ ]*[^ ]*[ ]*[^ ]* //' {} +
#sed -i 's/[ ]*[^ ]*[ ]*[^ ]* //' ${dir_labels}/mono_no_align/*.lab

for file_id in ${utts[@]}; do
    if [[ "${file_id}" == *\/* ]]; then
        speaker_id=${file_id%%/*}
#        echo $file_id $speaker_id
        if [ -n ${speaker_id} ]; then
            utt_id=${file_id##*/}
#            echo Copy ${utt_id} to ${speaker_id}/${utt_id}
            mkdir -p ${dir_labels}/{full,full_no_align,mono,mono_no_align}/${speaker_id}
            cp ${dir_labels}/full/${utt_id}.lab  ${dir_labels}/full/${speaker_id}/
            cp ${dir_labels}/full_no_align/${utt_id}.lab  ${dir_labels}/full_no_align/${speaker_id}/
            cp ${dir_labels}/mono/${utt_id}.lab  ${dir_labels}/mono/${speaker_id}/
            cp ${dir_labels}/mono_no_align/${utt_id}.lab  ${dir_labels}/mono_no_align/${speaker_id}/
        fi
    fi
done

echo "Generate MFCCs..."
./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/${name_file_id_list}_blockJOB.log \
        ${dir_misc}/alignment/state_align/gen_mfcc.py \
                --dir_wav ${dir_data}/wav/ \
                --dir_mfcc ${dir_mfcc} \
                --file_id_list ${dir_labels}/${name_file_id_list}_blockJOB
# Remove intermediate files.
eval rm -f ${dir_labels}/${name_file_id_list}_block{0..${num_blocks}}

# Force align labels.
echo "Force align labels..."
python ${dir_misc}/alignment/state_align/force_align.py \
            --dir_work ${dir_labels} \
            --dir_wav ${dir_data}/wav/ \
            --dir_lab ${dir_labels}/full_no_align/ \
            --dir_mfcc ${dir_mfcc} \
            --file_id_list ${file_id_list} \
            --dir_lab_align ${dir_labels}/label_state_align/ \
            ${multispeaker}

# Check number of files and identify samples where alignment failed.
num_aligned_labels=$(ls -1 ${dir_labels}/label_state_align/ | wc -l)
if [ "$num_utts" -ne "$num_aligned_labels" ]; then
    utts_aligned=()  # Create a new array with those utterances which were successfully aligned.
    utts_failed=()
    for file_id in ${utts[@]}; do
        utt_id=${file_id##*/}
        if [ ! -e "${dir_labels}/label_state_align/${utt_id}.lab" ]; then
            utts_failed+=("${utt_id}")
#            echo "ERROR: Alignment of ${utt_id} failed".
        else
            utts_aligned+=("${file_id}")
        fi
    done

    if [ ${#utts_failed[@]} -gt 0 ]; then
        echo "ERROR: Alignment of ${#utts_failed[@]}/${num_utts} samples failed."
        while true; do
            read -p "Do you want to list all failed samples? " yn
            case $yn in
                [Yy]* ) echo "${utts_failed[@]}"; break;;
                [Nn]* ) break;;
            * ) echo "Please answer yes or no.";;
            esac
        done

        # Offer the user a possibility to automatically correct the file_id_list.
        while true; do
            read -p "Do you want to correct the file_id_list at ${file_id_list} by excluding the failed samples? " yn
            case $yn in
                [Yy]* ) printf "%s\n" "${utts_aligned[@]}" > "${file_id_list}"; exit;;
                [Nn]* ) exit;;
            * ) echo "Please answer yes or no.";;
            esac
        done
    fi
fi
