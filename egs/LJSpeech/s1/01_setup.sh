#!/usr/bin/env bash
# Copyright 2018 by Idiap Research Institute, http://www.idiap.ch
#
# Author(s):
#   Bastian Schnell, 04.04.2018
#


usage() {
cat <<- EOF
    usage: ${PROGNAME} [OPTIONS] <Path to LJSpeech db> <num_workers=1>
    
    This program loads the samples from the LJSpeech database.
    It requires the path to the database as parameter.
    Ensure that bc and soxi packages are installed.

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

# Magic variables.
demo_set_size=300
target_frame_rate=16000
random_seed=42

# Seeding adopted from https://stackoverflow.com/a/41962458/7820599
# Seems to be the only way to get a seed into the shuf function.
get_seeded_random()
{
  seed="$1";
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null;
}

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
dir_audio="${dir_data}/wav/"
dir_logs="${dir_data}/log/"
dir_txt="${dir_data}/txt"

mkdir -p ${dir_exp}
mkdir -p ${dir_data}
mkdir -p ${dir_audio}
mkdir -p ${dir_txt}

file_id_list_demo="${dir_data}/file_id_list_demo.txt"
file_id_list="${dir_data}/file_id_list_full.txt"

# Collect utterance ids of audio files.
echo "Collect utterance ids of audio files..."
# Create the full set.
utts_full=$(ls "${db_path}"/wavs/)
utts_full="${utts_full//.wav/ }"  # Remove the wav extension.
utts_full=($utts_full)  # Convert to array.
printf "%s\n" "${utts_full[@]}" >| ${file_id_list}
# Create the demo set.
utts_full=( $(shuf --random-source=<(get_seeded_random $random_seed) -e ${utts_full[@]}))  # Shuffle array.
utts_demo=("${utts_full[@]:0:$demo_set_size}")
IFS=$'\n' utts_demo=($(sort <<<"${utts_demo[*]}"))  # Sort the demo set for readability.
unset IFS
printf "%s\n" "${utts_demo[@]}" >| ${file_id_list_demo}

utts_data=()
mapfile -t -O ${#utts_data[@]} utts_data < ${db_path}/metadata.csv # -t remove trailing newline, -O start index to add entries.
utts_data=("${utts_data[@]/\|*\|/ }")  # Substitute |<non-normalized text>| with whitespace.
printf "%s\n" "${utts_data[@]}" >| ${dir_data}/utts.data

# Create links to audio files.
echo "Create links to audio files..."
for utt in "${utts_full[@]}"; do
    # cp ${db_path}/wavs/${utt}.wav $dir_audio/${utt}.wav  # Copy files instead of symbolic link.
    ln -sf ${db_path}/wavs/${utt}.wav ${dir_audio}/${utt}.wav
done

# Down sampling.
echo "Downsample files to frame rate: ${target_frame_rate} ..."
rm -r -f "${dir_data}/wav_org"
mv "${dir_audio}" "${dir_data}/wav_org"
mkdir -p "${dir_audio}"

# Load utts list.
IFS=$'\r\n' GLOBIGNORE='*' command eval 'utts=($(cat $file_id_list))'
num_utts=${#utts[@]}
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
name_file_id_list=$(basename ${file_id_list})
split --numeric=1 --suffix-length ${suffix_length} -l ${block_size} ${file_id_list} ${dir_data}/${name_file_id_list}_block
# Remove leading zeros in block numbering.
for FILE in $(ls ${dir_data}/${name_file_id_list}_block*); do
    mv "${FILE}" "$(echo ${FILE} | sed -e 's:_block0*:_block:')" 2>/dev/null
done

./${cpu_1d_cmd} JOB=1:${num_blocks} ${dir_logs}/${name_file_id_list}_down_sampling_blockJOB.log \
                ${dir_src}/data_preparation/audio/down_sampling.py \
                ${dir_data}/wav_org/ \
                ${dir_data}/wav/ \
                ${dir_data}/${name_file_id_list}_blockJOB \
                ${target_frame_rate}

cp -R -u -p "${dir_data}/wav_org/*" "${dir_audio}"  # -u copy only when source is newer than destination file or if is missing, -p preserve mode, ownership, timestamps etc.
rm -r -f "${dir_data}"/wav_org/
eval rm -f ${dir_data}/${name_file_id_list}_down_sampling_block{0..${num_blocks}}  # eval command required because otherwise brace expansion { .. } happens before $ expansion
# Print errors and warnings.
eval grep --ignore-case "WARNING" "${dir_logs}/${name_file_id_list}_down_sampling_block{1..${num_blocks}}.log"
eval grep --ignore-case "ERROR" "${dir_logs}/${name_file_id_list}_down_sampling_block{1..${num_blocks}}.log"

if [ "$silence_removal" = true ]; then
    echo "Remove silence..."

    num_utts=${#utts_full[@]}
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
    name_file_id_list=$(basename ${file_id_list})
    split --numeric=1 --suffix-length ${suffix_length} -l ${block_size} ${file_id_list} ${dir_data}/${name_file_id_list}_block
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
    cp -R -u -p "${dir_data}/wav_org_silence/*" "${dir_data}/wav"  # -u copy only when source is newer than destination file or if is missing, -p preserve mode, ownership, timestamps etc.

    # Remove intermediate files.
    rm -r -f "${dir_data}"/wav_org_silence/
    eval rm -f ${dir_data}/${name_file_id_list}_block{0..${num_blocks}}  # eval command required because otherwise brace expansion { .. } happens before $ expansion

    # Print errors and warnings.
    eval grep --ignore-case "WARNING" "${dir_logs}/${name_file_id_list}_silence_removal_block{1..${num_blocks}}.log"
    eval grep --ignore-case "ERROR" "${dir_logs}/${name_file_id_list}_silence_removal_block{1..${num_blocks}}.log"
fi

if (( $(echo "$max_length_sec > 0" | bc -l) )); then  # bc returns 1 or 0, (( )) translates to true or false.
    echo "Removing audio files longer than ${max_length_sec} seconds..."

    # Remove too long files.
    # TODO: Call it on the grid. Move this into an explicit script?
    num_removed=0
    for filename in "${utts_full[@]}"; do
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
fi