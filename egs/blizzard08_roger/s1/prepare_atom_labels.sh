#!/usr/bin/env bash
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


usage() {
cat <<- EOF
    usage: $PROGNAME <voice_name=demo> <num_workers=1> <file_id_list=database/wcad_file_id_list_<voice_name>>
    
    Extract atoms for all ids listed in <file_id_list>.
    Audio files are searched in database/wav/.

    OPTIONS:
        -h                        show this help

    
    Examples:
        Run all ids with 10 jobs:
        $PROGNAME demo 10 file_id_list_demo.txt
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

function join_by { local IFS="$1"; shift; echo "$*"; }

# Fixed paths.
dir_data_prep=$(realpath "../../../idiaptts/src/data_preparation/")
dir_tools=$(realpath "../../../tools/")
dir_misc=$(realpath "../../../idiaptts/misc/")
dir_data=$(realpath "database/")

# Read parameters.
voice=${1:-"demo"}
num_workers=${2:-"1"}  # Default number of workers is one.
file_id_list=${3:-"${dir_data}/wcad_file_id_list_${voice}.txt"}
file_questions=${4:-"${dir_tools}/tts_frontend/questions/questions-en-radio_dnn_416.hed"}
#thetas=(0.015 0.033 0.058 0.087 0.17)
thetas=(0.030 0.045 0.060 0.075 0.090 0.105 0.120 0.135 0.150)
k=2

# Fixed path with parameters.
dir_audio=$(realpath "${dir_data}/wav/")
dir_out=$(realpath "experiments/${voice}/wcad-$(join_by '_' ${thetas[@]})/")
#dir_labels=$(realpath "experiments/${voice}/labels/label_state_align/")
#dir_questions=$(realpath "experiments/${voice}/questions/")
dir_logs="${dir_out}/log/"

# Create necessary directories.
mkdir -p "${dir_out}"
mkdir -p "${dir_logs}"

# Load utts list.
IFS=$'\r\n' GLOBIGNORE='*' command eval 'utts=($(cat $file_id_list))'
num_utts=${#utts[@]}
block_size=$(expr $num_utts / $num_workers + 1)
num_blocks=$(expr $num_utts / $block_size + 1)
echo "num_utts_train: " $num_utts
echo "num_workers: " $num_workers
echo "block_size: " $block_size
echo "num_blocks: " $num_blocks

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


echo "Generate atoms..."
./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/${name_file_id_list}_blockJOB.log \
        ${dir_data_prep}/wcad/AtomLabelGen.py \
                --wcad_root ${dir_tools}/wcad \
                --dir_audio ${dir_audio} \
                --dir_out ${dir_out} \
                --file_id_list ${dir_out}/${name_file_id_list}_blockJOB \
                -k ${k} \
                --thetas ${thetas[@]}


# Combine normalisation parameters of all blocks.
file_list_min_max=()
file_list_stats=()
for (( b=1; b <= $num_blocks; b++ )); do
    file_list_min_max+=("${dir_out}"/${name_file_id_list}_block${b}-min-max.bin)
    file_list_stats+=("${dir_out}"/${name_file_id_list}_block${b}-stats.bin)
done
python3 ${dir_misc}/normalisation/MinMaxExtractor.py \
                --dir_out "${dir_out}" \
                --file_list "${file_list_min_max[@]}"
python3 ${dir_misc}/normalisation/MeanStdDevExtractor.py \
                --dir_out "${dir_out}" \
                --file_list "${file_list_stats[@]}"

# Remove intermediate files.
for file in "${file_list_min_max[@]}" "${file_list_stats[@]}"; do
    rm "${file}"
done
for (( b=1; b <= $num_blocks; b++ )); do
    rm "${dir_out}"/${name_file_id_list}_block${b}-mean-std_dev.bin
done

# Print errors and save warnings.
eval grep --ignore-case "WARNING" "${dir_logs}/${name_file_id_list}_block{1..${num_blocks}}.log" >| "${dir_logs}/${name_file_id_list}_WARNINGS.txt"
eval grep --ignore-case "ERROR" "${dir_logs}/${name_file_id_list}_block{1..${num_blocks}}.log" | grep -v "0 with errors"

# Compute id lists with the ids for which atom extraction worked.
eval cat ${dir_data}/wcad_${name_file_id_list}_block{1..${num_blocks}}.txt | sort -t _ -k 2 -g > ${dir_data}/wcad_${name_file_id_list}.txt  # -t separator, -k key/column used for sorting, -g general numeric sort

# Remove intermediate files.
eval rm -f ${dir_out}/${name_file_id_list}_block{1..${num_blocks}}
eval rm -f ${dir_data}/wcad_${name_file_id_list}_block{1..${num_blocks}}.txt
