#!/usr/bin/env bash
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

usage() {
cat <<- EOF
    usage: $PROGNAME <voice_name=demo> <num_workers=1> <file_id_list=database/file_id_list_<voice_name>> <num_coded_sps=60>
    
    Extract WORLD features with deltas for all ids listed in <file_id_list>.
    Audio files are searched in database/wav/.

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
dir_data_prep=$(realpath "../../../src/data_preparation/")
dir_tools=$(realpath "../../../tools/")
dir_misc=$(realpath "../../../misc/")
dir_data=$(realpath "database/")
mkdir -p "experiments"
dir_experiments=$(realpath "experiments")

# Read parameters.
voice=${1:-"demo"}
num_workers=${2:-"1"}  # Default number of workers is one.
file_id_list=${3:-"${dir_data}/file_id_list_${voice}.txt"}
num_coded_sps=${4:-"60"}  # Default dimension of frequency features. Should depend on sampling frequency.
sp_type="mgc"

# Fixed path with parameters.
dir_audio="${dir_data}/wav/"
dir_out="${dir_experiments}/${voice}/WORLD/"
#dir_labels=$(realpath "${dir_experiments}/${voice}/labels/label_state_align/")
#dir_questions=$(realpath "${dir_experiments}/${voice}/questions/")
dir_logs="${dir_out}/log/"
dir_deltas="${dir_out}/cmp_${sp_type}${num_coded_sps}/"

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

echo "Generate WORLD deltas features..."
./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/${name_file_id_list}_blockJOB.log \
        ${dir_data_prep}/world/WorldFeatLabelGen.py \
                --dir_audio ${dir_audio} \
                --dir_out ${dir_out} \
                --file_id_list ${dir_out}/${name_file_id_list}_blockJOB \
                --add_deltas \
                --num_coded_sps ${num_coded_sps}

# Combine normalisation parameters of all blocks.
for feature in "lf0" "bap" "mgc${num_coded_sps}"; do
    #mkdir -p "${dir_out}"/${feature}/
    #file_list_min_max=()
    file_list_mean_covariance=()
    for (( b=1; b <= $num_blocks; b++ )); do
        #file_list_min_max+=("${dir_out}"/${name_file_id_list}_block${b}-min-max.bin)
        file_list_mean_covariance+=("${dir_deltas}"${name_file_id_list}_block${b}_${feature}-stats.bin)
    done

    echo "Combining stats for ${feature}..."
    python3 ${dir_misc}/normalisation/MeanCovarianceExtractor.py \
                    --dir_out "${dir_deltas}" \
                    --file_list "${file_list_mean_covariance[@]}" \
                    --file_name "_${feature}"

    # Remove intermediate files.
    for (( b=1; b <= $num_blocks; b++ )); do
        rm "${dir_deltas}"${name_file_id_list}_block${b}_${feature}-stats.bin
        rm "${dir_deltas}"${name_file_id_list}_block${b}_${feature}-mean-covariance.bin
    done
done

# Print errors and save warnings.
eval grep --ignore-case "WARNING" "${dir_logs}/${name_file_id_list}_block{1..${num_blocks}}.log"  # >| "${dir_logs}/${name_file_id_list}_WORLD_extraction_WARNINGS.txt"
eval grep --ignore-case "ERROR" "${dir_logs}/${name_file_id_list}_block{1..${num_blocks}}.log"

eval rm -f ${dir_out}/${name_file_id_list}_block{0..$num_blocks}
