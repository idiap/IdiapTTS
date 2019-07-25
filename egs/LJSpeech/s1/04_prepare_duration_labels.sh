#!/usr/bin/env bash

# Copyright 2018 by Idiap Research Institute, http://www.idiap.ch
#
# Author(s):
#   Bastian Schnell, 04.04.2018
#
# Execution requires: source cmd.sh


usage() {
cat <<- EOF
    usage: $PROGNAME <voice_name=demo> <num_workers=1> <file_id_list=database/file_id_list_<voice_name>>

    
    Extract durations for all ids listed in <file_id_list> from experiments/<voice_name>/labels/label_state_align/
    and saves them in experiments/<voice_name>/dur

    OPTIONS:
        -h                        show this help

    
    Examples:
        Run all ids with 10 jobs:
        $PROGNAME demo 10 file_id_list_full.txt
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
dir_data_prep=$(realpath "../../../idiaptts/src/data_preparation/")
dir_tools=$(realpath "../../../tools/")
dir_misc=$(realpath "../../../idiaptts/misc/")
dir_data=$(realpath "database/")
mkdir -p "experiments"
dir_experiments=$(realpath "experiments")

# Read parameters.
voice=${1:-"demo"}
num_workers=${2:-"1"}  # Default number of workers is one.
file_id_list=${3:-"${dir_data}/file_id_list_${voice}.txt"}

# Fixed path with parameters.
dir_out="${dir_experiments}/${voice}/dur/"
dir_labels="${dir_experiments}/${voice}/labels/label_state_align/"
dir_logs="${dir_out}/log/"

# Create necessary directories.
mkdir -p "${dir_out}"
mkdir -p "${dir_logs}"

# Load utts list.
IFS=$'\r\n' GLOBIGNORE='*' command eval 'utts=($(cat $file_id_list))'
num_utts=${#utts[@]}
block_size=$(expr $num_utts / $num_workers + 1)
num_blocks=$(expr $num_utts / $block_size + 1)
#echo $block_size
#echo $num_blocks

# Split into working blocks.
if [ "$num_blocks" -gt "99" ]; then
    suffix_length=3
elif [ "$num_blocks" -gt "9" ]; then
    suffix_length=2
else 
    suffix_length=1
fi

# Split file_id_list.
name_file_id_list=$(basename "${file_id_list%.*}")
split --numeric=1 --suffix-length $suffix_length -l $block_size ${file_id_list} ${dir_out}/${name_file_id_list}_block
# Remove leading zeros in block numbering.
for FILE in $(ls ${dir_out}/${name_file_id_list}_block*); do
    mv "${FILE}" "$(echo ${FILE} | sed -e 's:_block0*:_block:')" 2>/dev/null
done


echo "Generate durations..."
./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/${name_file_id_list}_blockJOB.log \
        ${dir_data_prep}/phonemes/PhonemeDurationLabelGen.py \
                --dir_labels ${dir_labels} \
                --dir_out ${dir_out} \
                --file_id_list ${dir_out}/${name_file_id_list}_blockJOB


# Combine normalisation parameters of all blocks.
file_list_mean_std=()
for (( b=1; b <= $num_blocks; b++ )); do
    #file_list_min_max+=("${dir_out}"/${name_file_id_list}_block${b}-min-max.bin)
    file_list_mean_std+=("${dir_out}"/${name_file_id_list}_block${b}-stats.bin)
done
python3 ${dir_misc}/normalisation/MeanStdDevExtractor.py \
                --dir_out "${dir_out}" \
                --file_list "${file_list_mean_std[@]}"

# Remove intermediate files.
for (( b=1; b <= $num_blocks; b++ )); do
    rm "${dir_out}"/${name_file_id_list}_block${b}-stats.bin
    rm "${dir_out}"/${name_file_id_list}_block${b}-mean-std_dev.bin
done

eval rm -f ${dir_out}/${name_file_id_list}_block{0..$num_blocks}
