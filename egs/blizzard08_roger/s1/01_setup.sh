#!/usr/bin/env bash
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


usage() {
cat <<- EOF
    usage: ${PROGNAME} [OPTIONS] <Path to roger db> <num_workers=1>
    
    This program loads the samples from the roger database.
    It requires the path to roger database as parameter.
    Ensure that bc and soxi packages are installed.
    It creates a file_id_list_all.txt with all ids,
    a file_id_list_full.txt with the ids from carroll, arctic, and theherald1-3,
    and a file_id_list_demo.txt with the ids from theherald1.

    OPTIONS:
        -h                        show this help
        --no_silence_removal      skips the silence removal step
        --max_length_sec          maximum length of audio files used

EOF
}

die () {
    echo -e >&2 "ERROR" "$@"
    exit 1
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
dir_src=$(realpath "../../../idiaptts/src/")

# Parameter extraction.
silence_removal=true # Default parameter.
max_length_sec=-1
while getopts ":h-:" flag; do # If a character is followed by a colon (e.g. f:), that option is expected to have an argument.
        case "${flag}" in
            -) case "${OPTARG}" in
                   no_silence_removal) silence_removal=false ;;
                   max_length_sec) val="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                                   max_length_sec=${val} ;;
                   *) die "Invalid option: --${OPTARG}" ;;
               esac;;
            h) usage; exit ;;
            \?) die "Invalid option: -$OPTARG" ;;
            :)  die "Option -$OPTARG requires an argument." ;;
        esac
    done
    shift $(($OPTIND - 1)) # Skip the already processed arguments.

minArgs=1
if [[ $# -lt "${minArgs}" ]]; then
    usage # Function call.
    die "Wrong number of parameters, expected at least ${minArgs} but got $#."
fi
db_path=${1:-}
num_workers=${2:-"1"}  # Default number of workers is one.

dir_exp="${PROGDIR}/experiments"
dir_data="${PROGDIR}/database"
dir_audio="${dir_data}/wav"
dir_logs="${dir_data}/log/"
dir_txt="${dir_data}/txt"

mkdir -p ${dir_exp}
mkdir -p ${dir_data}
mkdir -p ${dir_audio}
mkdir -p ${dir_txt}

file_id_list_demo="${dir_data}/file_id_list_demo.txt"
file_id_list="${dir_data}/file_id_list_full.txt"
file_id_list_all="${dir_data}/file_id_list_all.txt"

# Collect utterance ids of audio files.
echo "Collect utterance ids of audio files..."
utt_lists_demo=("theherald1")
utts_demo=()
for utt_list in "${utt_lists_demo[@]}"; do
    mapfile -t -O ${#utts_demo[@]} utts_demo < ${db_path}/stp/${utt_list} # -t remove trailing newline, -O start index to add entries.
done
# Remove duplicates.
utts_demo=($(printf "%s\n" "${utts_demo[@]}" | sort -u))
printf "%s\n" "${utts_demo[@]}" >| ${file_id_list_demo}

utt_lists_full=("carroll" "arctic" "theherald")
utts_full=()
for utt_list in "${utt_lists_full[@]}"; do
    mapfile -t -O ${#utts_full[@]} utts_full < ${db_path}/stp/${utt_list} # -t remove trailing newline, -O start index to add entries.
done
# Remove duplicates.
utts_full=($(printf "%s\n" "${utts_full[@]}" | sort -u))
printf "%s\n" "${utts_full[@]}" >| ${file_id_list}

utt_list_all=("all")
utts_all=()
for utt_list in "${utt_list_all[@]}"; do
    mapfile -t -O ${#utts_all[@]} utts_all < ${db_path}/stp/${utt_list} # -t remove trailing newline, -O start index to add entries.
done
# Remove duplicates.
utts_all=($(printf "%s\n" "${utts_all[@]}" | sort -u))
printf "%s\n" "${utts_all[@]}" >| ${file_id_list_all}

# Create links to audio files.
echo "Create links to audio files..."
for utt in "${utts_all[@]}"; do
    # cp ${db_path}/wav/${utt:0:7}/${utt}.wav $dir_audio/${utt}.wav  # Copy files instead of symbolic link.
    ln -sf ${db_path}/wav/${utt:0:7}/${utt}.wav ${dir_audio}/${utt}.wav
done

if [ "$silence_removal" = true ]; then
    echo "Remove silence..."

    num_utts=${#utts_all[@]}
    block_size=$(expr ${num_utts} / ${num_workers} + 1)
    num_blocks=$(expr ${num_utts} / ${block_size} + 1)
    # Split into working blocks.
    if [ "$num_blocks" -gt "99" ]; then
        suffix_length=3
    elif [ "$num_blocks" -gt "9" ]; then
        suffix_length=2
    else
        suffix_length=1
    fi
    # Split file_id_list in the same way as utts_selected.data.
    name_file_id_list=$(basename ${file_id_list_all})
    split --numeric=1 --suffix-length ${suffix_length} -l ${block_size} ${file_id_list_all} ${dir_data}/${name_file_id_list}_block
    # Remove leading zeros in block numbering.
    for FILE in $(ls ${dir_data}/${name_file_id_list}_block*); do
        mv "${FILE}" "$(echo ${FILE} | sed -e 's:_block0*:_block:')" 2>/dev/null
    done

    rm -r -f "${dir_data}"/wav_org_silence/
    mv "${dir_data}"/wav "${dir_data}"/wav_org_silence
    mkdir -p "${dir_data}"/wav

    # python ${dir_src}/data_preparation/audio/silence_remove.py --dir_wav "database/wav_org_silence/" --dir_out "database/wav/" --file_id_list "database/file_id_list_full.txt"

    ./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/${name_file_id_list}_silence_removal_blockJOB.log \
        ${dir_src}/data_preparation/audio/silence_remove.py \
                --dir_wav ${dir_data}/wav_org_silence/ \
                --dir_out ${dir_data}/wav/ \
                --file_id_list ${dir_data}/${name_file_id_list}_blockJOB

    # Copy files not touched in this remove silence step.
    cp -R -u -p "${dir_data}/wav_org_silence/*" "${dir_data}/wav"  # -u copy only when source is newer than destination file or if it is missing, -p preserve mode, ownership, timestamps etc.

    # Remove intermediate files.
    rm -r -f "${dir_data}"/wav_org_silence/
    eval rm -f ${dir_data}/${name_file_id_list}_block{0..${num_blocks}}  # eval command required because otherwise brace expansion { .. } happens before $ expansion
fi

if (( $(echo "$max_length_sec > 0" | bc -l) )); then  # bc returns 1 or 0, (( )) translates to true or false.
    echo "Removing audio files longer than ${max_length_sec} seconds..."

    # Remove too long files.
    # TODO: Call it on the grid. Move this into an explicit script?
    num_removed=0
    for filename in "${utts_all[@]}"; do
        length=$(soxi -D "${dir_audio}/${filename}.wav")
        if (( $(echo "${length} > ${max_length_sec}" | bc -l) )); then
            rm "${dir_audio}/${filename}.wav"
            ((num_removed++))
        fi
    done
    echo "Removed ${num_removed} files."

    # Update file id lists.
    comm -12 <(printf '%s\n' "${utts_full[@]}") <(ls "${dir_audio}"/ | sed -e 's/\..*$//' | sort) > "${file_id_list}"
    comm -12 <(printf '%s\n' "${utts_demo[@]}") <(ls "${dir_audio}"/ | sed -e 's/\..*$//' | sort) > "${file_id_list_demo}"
    comm -12 <(printf '%s\n' "${utts_all[@]}") <(ls "${dir_audio}"/ | sed -e 's/\..*$//' | sort) > "${file_id_list_all}"
fi

# Copy labels file to data directory.
cp ${db_path}/utts.data ${dir_data}/

#echo "Create utts_selected.data file containing only the utterance of the subset."
## Combine the selected utterances to a regex pattern.
#utts_pat=$(echo ${utts_full[@]}|tr " " "|")
## Select those labes of utts.data which belong to the selected utterances.
#cat ${dir_data}/utts.data | grep -wE "${utts_pat}" >| ${dir_txt}/utts_selected.data

## Turn every line of utts.data into a txt file using the utterance id as file name.
#echo "Create txt file with label for each utterance in database/txt/..."
#awk -F' ' -v outDir=${dir_txt} '{print substr($0,length($1)+2,length($0)) > outDir"/"substr($1,2,length($1)-1)".txt"}' ${dir_txt}/utts_selected.data

# Remove intermediate files.
#rm ${dir_txt}/utts_selected.data
